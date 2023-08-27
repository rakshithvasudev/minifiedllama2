#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

// Transformer model setup

typedef struct {

  int dim;
  int hidden_dim;
  int n_layers;
  int n_heads;
  int n_kv_heads;
  int vocab_size; // vocab size usually 256 byte level
  int seq_len;    // max seq len

} Config;

typedef struct {

  // token embedding table
  float *token_embedding_table; // (vocab_size, dim)

  // weight for rmsnorms
  float *rms_att_weight; // (layer, dim) rmsnorm weights

  float *rms_ffn_weight; // (layer, dim)

  // weights for matmuls note: dim == n_heads * head_size
  float *wq; // (layer, dim, n_heads * head_size)

  float *wk; // (layer, dim, n_kv_heads * head_size)

  float *wv; // (layer, dim, n_kv_heads * head_size)

  float *wo; // (later, n_heads * head_size, dim)

  // weights for ffn

  float *w1; // (layer, hidden_dim, dim)

  float *w2; // (layer, dim, hidden_dim)

  float *w3; // (layer, hidden_dim, dim)

  // final rmsnorm

  float *rms_final_weight; // (dim,)

  float *wcls;

} TransformerWeights;

typedef struct {

  // current wave of activations
  float *x;      // activation at current timestamp (dim,)
  float *xb;     // same, but inside residual branch (dim, )
  float *xb2;    // additional buffer for convenience (dim, )
  float *hb;     // buffer for hidden dim in ffn (hidden_dim, )
  float *hb2;    // buffer for hidden dim in ffn (hidden_dim, )
  float *q;      // query (dim, )
  float *k;      // key (dim, )
  float *v;      // value (dim, )
  float *att;    // buffer for scores/attention values (seq_len, )
  float *logits; // output logits
  // kv cache
  float *key_cache;   // (layer, seq_len, dim)
  float *value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
  Config config; // the hyperparameters of the architecture (the blueprint)
  TransformerWeights weights; // the weights of the model
  RunState state; // buffers for the "wave" of activations in the forward pass
  // some more state needed to properly clean up the memory mapping (sigh)
  int fd;            // file descriptor for memory mapping
  float *data;       // memory mapped data pointer
  ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState *s, Config *p) {

  // we calloc instead of malloc to keep valgrind working with us and support us
  s->x = calloc(p->dim, sizeof(float));
  s->xb = calloc(p->dim, sizeof(float));
  s->xb2 = calloc(p->dim, sizeof(float));
  s->hb = calloc(p->hidden_dim, sizeof(float));
  s->hb2 = calloc(p->hidden_dim, sizeof(float));
  s->q = calloc(p->dim, sizeof(float));
  s->k = calloc(p->dim, sizeof(float));
  s->v = calloc(p->dim, sizeof(float));
  s->att = calloc(p->seq_len, sizeof(float));
  s->logits = calloc(p->vocab_size, sizeof(float));
  s->key_cache = calloc(p->n_layers * p->seq_len * p->dim, sizeof(float));
  s->value_cache = calloc(p->n_layers * p->seq_len * p->dim, sizeof(float));

  // ensure all callocs went fine

  if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k ||
      !s->v || !s->att || !s->logits || !s->key_cache || !s->value_cache) {
    fprintf(stderr, "malloc failed!\n");
    exit(EXIT_FAILURE);
  }
}

void free_run_state(RunState *s) {
  free(s->x);
  free(s->xb);
  free(s->xb2);
  free(s->hb);
  free(s->hb2);
  free(s->q);
  free(s->k);
  free(s->v);
  free(s->att);
  free(s->logits);
  free(s->key_cache);
  free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config *p, float *ptr,
                        int shared_weights) {
  int head_size = p->dim / p->n_heads;
  w->token_embedding_table = ptr;
  ptr += p->vocab_size * p->dim;
  w->rms_att_weight = ptr;
  ptr += p->n_layers * p->dim;
  w->wq = ptr;
  ptr += p->n_layers * p->dim * (p->n_heads * head_size);
  w->wk = ptr;
  ptr += p->n_layers * p->dim * (p->n_kv_heads * head_size);
  w->wv = ptr;
  ptr += p->n_layers * p->dim * (p->n_kv_heads * head_size);
  w->wo = ptr;
  ptr += p->n_layers * p->dim * (p->n_heads * head_size) * p->dim;
  w->rms_ffn_weight = ptr;
  ptr += p->n_layers * p->dim;
  w->w1 = ptr;
  ptr += p->n_layers * p->dim * p->hidden_dim;
  w->w2 = ptr;
  ptr += p->n_layers * p->hidden_dim * p->dim;
  w->w3 = ptr;
  ptr += p->n_layers * p->dim * p->hidden_dim;
  w->rms_final_weight = ptr;
  ptr += p->dim;
  ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real(ROPE)
  ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag(ROPE)
  w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char *checkpoint, Config *config,
                     TransformerWeights *weights, int *fd, float **data,
                     ssize_t *file_size) {

  FILE *file = fopen(checkpoint, "rb");

  if (!file) {
    fprintf(stderr, "Couldn't open file %s\n", checkpoint);
    exit(EXIT_FAILURE);
  }
  // read in the config header
  if (fread(config, sizeof(Config), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  // negative vocab size is hacky way of signaling unshared weights. bit yikes.
  int shared_weights = config->vocab_size > 0 ? 1 : 0;
  config->vocab_size = abs(config->vocab_size);
  // figure out the file size
  fseek(file, 0, SEEK_END); // move the file pointer to EOF
  *file_size = ftell(file); // get the file size in bytes
  fclose(file);

  // memory map the transformer weights into data pointer
  *fd = open(checkpoint, O_RDONLY); // open only in read mode
  if (*fd == -1) {
    fprintf(stderr, "open failed! \n");
    exit(EXIT_FAILURE);
  }

  *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (*data == MAP_FAILED) {
    fprintf(stderr, "mmap failed!\n");
    exit(EXIT_FAILURE);
  }

  float *weights_ptr = *data + sizeof(Config) / sizeof(float);
  memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char *checkpoint_path) {

  // read configs and weights from ckpt
  read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data,
                  &t->file_size);
  // allocate the Runstate buffers
  malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer *t) {
  // close the mmap
  if (t->data != MAP_FAILED) {
    munmap(t->data, t->file_size);
  }
  if (t->fd != -1) {
    close(t->fd);
  }
  // free the runstate buffers
  free_run_state(&t->state);
}

int main() { return 0; }
