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

/*
---------------------- Setting up essential NN blocks ---------------------
*/

void rmsnorm(float *o, float *x, float *weight, int size) {

  // sum of squares
  float ss = 0.0f;
  for (int j = 0; j < size; j++) {
    ss += x[j] * x[j];
  }

  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);

  // normalize and scale
  for (int j = 0; j < size; j++) {
    o[j] = weight[j] * (ss * x[j]);
  }
}

void softmax(float *x, int size) {

  // find max value for numerical stability
  float max_val = x[0];
  for (int i = 0; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }
}

void matmul(float *xout, float *x, float *w, int n, int d) {
  // W (d,n) @ x(n, ) -> xout (d, )
  // most time spent here
  int i;

#pragma omp parallel for private(i)
  for (i = 0; i < d; i++) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += w[i * n + j] * x[j];
    }
    xout[i] = val;
  }
}

float *forward(Transformer *transformer, int token, int pos) {

  // convienence variables
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;
  float *x = s->x;
  int dim = p->dim;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int kv_mul =
      (p->n_heads /
       p->n_kv_heads); // int multiplier of the kv sharing in multiquery
  int hidden_dim = p->hidden_dim;
  int head_size = dim / p->n_heads;

  // copy the token embedding to x
  float *content_row = w->token_embedding_table + token * dim;
  memcpy(x, content_row, dim * sizeof(*x));

  // forward all layers
  for (unsigned long long l = 0; l < p->n_layers; l++) {
    
    // attention rmsnorm
    rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

    // qkv matmuls for this position
    matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
    matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
    matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

    // RoPE relative pos encoding: complex-vlaued rotate q and k in each head
    for (int i = 0; i < dim; i += 2) {
      int head_dim = i % head_size;
      float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
      float val = pos * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1=q only
      for (int v = 0; v < rotn; v++) {
        float *vec =
            v == 0 ? s->q : s->k; // the vector to rotate (query or key)
        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
      }
    }
    // save key, value at this time step (pos) to our kv cache
    int loff = l * p->seq_len * kv_dim; // kv cache later offset for convenience
    float *key_cache_row = s->key_cache + loff + pos * kv_dim;
    float *value_cache_row = s->key_cache + loff + pos * kv_dim;
    memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
    memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

    // multihead attn
    int h;
    #pragma omp parallel for private(h)
    for (h = 0; h < p->n_heads; h++) {
      // get the query vector for this head
      float *q = s->q + h * head_size;
      // attention scores for this head
      float *att = s->att + h * p->seq_len;
      // iterate over all timestamps, including the current one
      for (int t = 0; t <= pos; t++) {
        // get the key vector for this head and at this timestamp
        float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        // calcuate the attention score as the dot product of q and k
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) {
          score += q[i] * k[i];
        }
        score /= sqrtf(head_size);
        // save the score to attention buffer
        att[t] = score;
      }

      // softmax the scores to get attention weights, from 0..pos inclusively
      softmax(att, pos + 1);

      // weighted sum of values, store back into xb
      float *xb = s->xb + h * head_size;
      memset(xb, 0, head_size * sizeof(float));
      for (int t = 0; t <= pos; t++) {
        // get the value vector for this head and at this timestep
        float *v =
            s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        // get the attention weight for this timestep
        float a = att[t];
        // accumulate the weighted value into xb
        for (int i = 0; i < head_size; i++) {
          xb[i] += a * v[i];
        }
      }
    }

    // final matmul to get the output of attention
    matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

    // residual connection back into x
    for (int i =0; i < dim; i++){
    x[i] += s->xb2[i];
   } 

    //ffn rms
    rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

    // Now for FFN in Pytorch we have: self.w2(F.silu(self.w1(x))*self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
    matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);
    
    // SwiGLU non linearity 

 


  int main() { return 0; }
