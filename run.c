/*

Setting up llama2 on commodity systems with a decent performance.
All inference only code setup. Simple C Inference. Repeat of
Andrej's llama2.c Inference, picking up pieces from
GGML. Manually typed to play around and
see if there could be any performance gains that
can be realized.

*/

#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#if defined _WIN32
#include "win.h"
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

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
    for (int i = 0; i < dim; i++) {
      x[i] += s->xb2[i];
    }

    // ffn rms
    rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

    // Now for FFN in Pytorch we have: self.w2(F.silu(self.w1(x))*self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
    matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

    // SwiGLU non linearity
    for (int i = 0; i < hidden_dim; i++) {
      float val = s->hb[i];
      // silu(x) = x * sigma(x), where sigma(x) is logistic sigmoid
      val *= (1.0f / (1.0f + expf(-val)));
      // elementwise multiply with w3(x)
      val *= s->hb2[i];
      s->hb[i] = val;
    }

    // final matmul to get the output of the ffn
    matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

    // residual connection
    for (int i = 0; i < dim; i++) {
      x[i] += s->xb[i];
    }
  }

  // final rmsnorm
  rmsnorm(x, x, w->rms_final_weight, dim);

  // classify into logits
  matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
  return s->logits;
}

// ----------------------------------------------------------------------
// BPE encoding tokenizer translation setup strings -> tokens and vice versa
typedef struct {
  char *str;
  int id;
} TokenIndex;

typedef struct {
  char **vocab;
  float *vocab_scores;
  TokenIndex *sorted_vocab;
  int vocab_size;
  unsigned int max_token_length;
  unsigned char byte_pieces[512];
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
  return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size) {

  // should have written vocab_size into the tokenizer file
  t->vocab_size = vocab_size;
  // malloc space to hold the scores and the strings
  t->vocab = (char **)malloc(vocab_size * sizeof(char *));
  t->vocab_scores = (float *)malloc(vocab_size * sizeof(char *));
  t->sorted_vocab = NULL; // lazy init
  for (int i = 0; i < 256; i++) {
    t->byte_pieces[i * 2] = (unsigned char)i;
    t->byte_pieces[i * 2 + 1] = '\0';
  }

  // read in the file
  FILE *file = fopen(tokenizer_path, "rb");
  if (!file) {
    fprintf(stderr, "Couldn't load %s \n", tokenizer_path);
    exit(EXIT_FAILURE);
  }
  if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {
    fprintf(stderr, "failed read\n");
    exit(EXIT_FAILURE);
  }
  int len;
  for (int i = 0; i < vocab_size; i++) {
    if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    if (fread(&len, sizeof(int), 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    t->vocab[i] = (char *)malloc(len + 1);
    if (fread(t->vocab[i], len, 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    t->vocab[i][len] = '\0'; // add the string terminating token
  }
  fclose(file);
}

void free_tokenizer(Tokenizer *t) {

  for (int i = 0; i < t->vocab_size; i++) {
    free(t->vocab[i]);
  }
  free(t->vocab);
  free(t->vocab_scores);
  free(t->sorted_vocab);
}

char *decode(Tokenizer *t, int prev_token, int token) {

  char *piece = t->vocab[token];
  // following BOS (1) token, sentencepiece decoder strips any leading
  // whitespace
  if (prev_token == 1 && piece[0] == ' ') {
    piece++;
  }
  // careful, some tokens designate raw bytes and look like ex: <0x01>
  // parse this and convert and return the actual byte
  unsigned char byte_val;
  if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
    piece = (char *)t->byte_pieces + byte_val * 2;
  }
  return piece;
}

void safe_printf(char *piece) {
  // piece might be a raw byte token, and we only want to print printable chars
  // or whitespace because some of the other bytes can be various control codes,
  // backspace, etc.
  if (piece == NULL) {
    return;
  }
  if (piece[0] == '\0') {
    return;
  }
  if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val))) {
      return; // bad byte, don't print it
    }
  }
  printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
  // efficiently find the perfect match for str in vocab, return its index or -1
  // if not found
  TokenIndex tok = {.str = str}; // acts as the key to search for
  TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex),
                            compare_tokens);
  return res != NULL ? res->id : -1;
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens,
            int *n_tokens) {
  // ecnode the string text input into an upper bound preallocated tokens[]
  // array bos != 0 means prepend BOS token (=1), eos != 0 means append the EOS
  // token (=2)
  if (text == NULL) {
    fprintf(stderr, "Cannot encode NULL Text \n");
    exit(EXIT_FAILURE);
  }

  if (t->sorted_vocab == NULL) {
    // lazy malloc and sort vocab
    t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < t->vocab_size; i++) {
      t->sorted_vocab[i].str = t->vocab[i];
      t->sorted_vocab[i].id = i;
    }
    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
  }

  // create a temp buffer that will store merge candidates of always 2
  // consequtive tokens 2* for concat, +1 for null terminator +2 for UTF8 (in
  // case max_token_length is 1)
  char *str_buffer = malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
  size_t str_len = 0;

  // start at 0 tokens
  *n_tokens = 0;

  // add optional BOS (=1) token if needed
  if (bos)
    tokens[(*n_tokens)++] = 1;

  // add_dummy_prefix is true by default
  // so prepend a dummy prefix token to the input string, but only if text!= ""
  // TODO: pretty sure this isn't correct in general case, more to be read from
  // sentencepiece to find out what's going on
  if (text[0] != '\0') {
    int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
    tokens[(*n_tokens)++] = dummy_prefix;
  }

  // UTF-8 Time and its messy. Reference from Wikipedia:
  // code point <-> UTF8 conversion
  // First code point last code point Byte1 Byte2 Byte3 Byte4
  // U+0000	U+007F	    0xxxxxxx
  // U+0080	U+07FF	    110xxxxx	10xxxxxx
  // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
  // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

  // process the raw UTF-8 bye sequence of input string
  for (char *c = text; *c != '\0'; c++) {
    // reset buffer if the current byte is ASCII or a leading byte
    // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the
    // rest 0x80 is 10000000 in UTF-8, all continuation bytes start with "10" in
    // first two bits so in English this is: "if this byte is not a continuation
    // byte"
    if ((*c & 0xC0) != 0x80) {
      // this byte must be either a leading byte (11...) or an ASCII char
      // (0x...)
      // => reset our location, as we're starting a new UTF-8 codepoint
      str_len = 0;
    }

    // append the current byte to the buffer
    str_buffer[str_len++] =
        *c; // ++ is post increment, incremented after this line
    str_buffer[str_len] = '\0';

    // while the next character is a continuation byte, continue appending
    // but if there are too many of them, just stop to avoid overrunning
    // str_buffer size.
    if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
      continue;
    }

    // ok c+1 isn't a continuation byte, so we've read in full codepoint
    int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

    if (id != -1) {
      // we found this codepoint in vocab, add it as a token
      tokens[(*n_tokens)++] = id;
    } else {
      // byte_fallback encoding: just encode each byte as a token
      // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
      // so the individual btes only start at index 3
      for (int i = 0; i < str_len; i++) {
        tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
      }
    }

    str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
  }

  // merge the best consecutive pair each iteration, according to scores in
  // vocab_scores

  while (1) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;

    for (int i = 0; i < (*n_tokens - 1); i++) {
      // check if we can merge the pair (tokens[i], tokens[i+1])
      sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
      int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
      if (id != -1 && t->vocab_scores[id] > best_score) {
        // this merge pair exists in vocab! record its score and position
        best_score = t->vocab_scores[id];
        best_id = id;
        best_idx = i;
      }
    }

    if (best_idx == -1) {

      break; // couldn't find anymore pairs to merge, so we're done
    }

    // merge the consecutive pair (best_idx, best_idx +1) into new token best_id
    tokens[best_idx] = best_id;
    // delete token at position best_idx + 1, shift the entire sequence back 1
    for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
      tokens[i] = tokens[i + 1];
    }
    (*n_tokens)--; // token length decreased
  }

  // add optional EOS (=2) token if desired
  if (eos)
    tokens[(*n_tokens)++] = 2;

  free(str_buffer);
}

// ---------------------------------------------------------------------------------
// Sampler which takes logits and returns a sampled token
// sampling can be done in few ways: greedy argmax sampling, top-p sampling

typedef struct {
  float prob;
  int index;
} ProbIndex; // struct for sorting probabilities during top-p sampling

typedef struct {
  int vocab_size;
  ProbIndex *probindex; // buffer used in top-p sampling
  float temperature;
  float topp;
  unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probabilities, int n) {

  // return the index that has the highest probability
  int max_i = 0;
  float max_p = probabilities[0];
  for (int i = 1; i < n; i++) {
    if (probabilities[i] > max_p) {
      max_i = i;
      max_p = probabilities[i];
    }
  }

  return max_i;
}

int sample_multi(float *probabilites, int n, float coin) {
  // sample index from probabilities (they must sum to 1)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilites[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}

int compare(const void *a, const void *b) {
  ProbIndex *a_ = (ProbIndex *)a;
  ProbIndex *b_ = (ProbIndex *)b;

  if (a_->prob > b_->prob)
    return -1;
  if (a_->prob < b_->prob)
    return 1;

  return 0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex,
                float coin) {
  // top p sampling (or nucleus sampling) samples from the smallest set of
  // tokens that exceed probability topp. This way we never sample tokens that
  // have very low probabilities are less likely to go "off the rails" coin is
  // random number in [0,1) usually from random_f32()

  int n0 = 0;
  // quicksort indices in desc order of probabilities
  // values smaller than (1-topp)/(n-1) cannot be a part of the result
  // so for efficiency we crop these out as candidates before sorting
  const float cutoff = (1.0f - topp) / (n - 1);
  for (int i = 0; i < n; i++) {
    if (probabilities[i] >= cutoff) {
      probindex[n0].index = i;
      probindex[n0].prob = probabilities[i];
      n0++;
    }
  }

  qsort(probindex, n0, sizeof(ProbIndex), compare);

  // truncate the list where cumulative probability exceeds topp
  float cumulative_prob = 0.0f;
  int last_idx = n0 - 1; // incase of rounding errors consider all elements
  for (int i = 0; i < n0; i++) {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > topp) {
      last_idx = i;
      break; // we've exceeded topp by including last_idx
    }
  }

  // sample from truncated list
  float r = coin * cumulative_prob;
  float cdf = 0.0f;
  for (int i = 0; i <= last_idx; i++) {
    cdf += probindex[i].prob;
    if (r < cdf) {
      return probindex[i].index;
    }
  }

  return probindex[last_idx].index; // incase of rounding errors
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature,
                   float topp, unsigned long long rng_seed) {

  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  // buffer only used with nucleus sampling; may not need but its small
  sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler) { free(sampler->probindex); }

unsigned int random_u32(unsigned long long *state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) {
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler *sampler, float *logits) {

  // sample the token given the logits and some hparams
  int next;
  if (sampler->temperature == 0.0f) {
    // greedy argmax sampling: take the token with highest probs
    next = sample_argmax(logits, sampler->vocab_size);
  } else {
    // apply the temperature to the logits
    for (int q = 0; q < sampler->vocab_size; q++) {
      logits[q] /= sampler->temperature;
    }
    // apply softmax to logits to get the probs for the next token
    softmax(logits, sampler->vocab_size);
    // flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&sampler->rng_state);
    // we sample from this distribution to get the next token
    if (sampler->topp <= 0 || sampler->topp >= 1) {
      // simply sample from the predicted probability distribution
      next = sample_multi(logits, sampler->vocab_size, coin);
    } else {
      // top -p (nucleus) sampling, clamping the least likely tokens to zero
      next = sample_topp(logits, sampler->vocab_size, sampler->topp,
                         sampler->probindex, coin);
    }
  }
  return next;
}


// ------------------------------------------------------------------------------------
// utilities time

long time_in_ms() {
// return time in milliseconds, for benchmarking the model speed
struct timespec time;
clock_gettime(CLOCK_REALTIME, &time);
return time.tv_sec * 1000 + time.tv_nsec/100000;
}

int main() { return 0; }
