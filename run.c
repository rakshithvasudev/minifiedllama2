#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>


// Transformer model setup

typedef struct {

int dim;
int hidden_dim; 
int n_layers;
int n_heads;
int n_kv_heads;
int vocab_sizes; // vocab size usually 256 byte level
int seq_len; // max seq len

} Config;


typedef struct {

// token embedding table 
float* token_embedding_table; // (vocab_size, dim)

// weight for rmsnorms
float* rms_att_weight; // (layer, dim) rmsnorm weights

float* rms_ffn_weight; // (layer, dim)

// weights for matmuls note: dim == n_heads * head_size 
float* wq; // (layer, dim, n_heads * head_size) 

float* wk; // (layer, dim, n_kv_heads * head_size) 

float* wv; // (layer, dim, n_kv_heads * head_size) 

float* wo; // (later, n_heads * head_size, dim)

// weights for ffn

float* w1; // (layer, hidden_dim, dim)

float* w2; // (layer, dim, hidden_dim)

float* w3; // (layer, hidden_dim, dim)

// final rmsnorm 

float* rms_final_weight; // (dim,)

float* wcls;

} TransformerWeights;


typedef struct {

// current wave of activations
float *x; // activation at current timestamp (dim,)
float *xb; // same, but inside residual branch (dim, )
float *xb2; // additional buffer for convenience (dim, )
float *hb; // buffer for hidden dim in ffn (hidden_dim, )
float *hb2; // buffer for hidden dim in ffn (hidden_dim, )
float *q; // query (dim, )
float *k; // key (dim, )
float *v; // value (dim, )
float *att; // buffer for scores/attention values (seq_len, )
float *logits; // output logits
// kv cache 
float* key_cache; // (layer, seq_len, dim)
float* value_cache; // (layer, seq_len, dim)
} RunState;



void malloc_run_state(RunState* s, Config* p){

// we calloc instead of malloc to keep valgrind working with us and support us 
s -> x = calloc(p->dim, sizeof(float));
s -> xb = calloc(p -> dim, sizeof(float));
s -> xb2 = calloc(p -> dim, sizeof(float));
s -> hb = calloc(p -> hidden_dim, sizeof(float));
s -> hb2 = calloc(p -> hidden_dim, sizeof(float));
s -> q = calloc(p -> dim, sizeof(float));
s -> k = calloc(p -> dim, sizeof(float));
s -> v = calloc(p -> dim, sizeof(float));
s -> att = calloc(p -> seq_len, sizeof(float));
s -> logits = calloc(p -> vocab_size, sizeof(float));
s -> key_cache = calloc(p -> n_layers * p->seq_len * p->dim, sizeof(float));
s -> value_cache = calloc(p -> n_layers * p->seq_len * p->dim, sizeof(float));

// ensure all callocs went fine

if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q 
 || !s->k || !s->v || !s->att || !s->logits || !s->key_cache 
 || !s->value_cache) {
    printf("malloc failed!\n");
    exit(1);
}


void free_run_state(RunState* s) {
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




int main(){

return 0;

}


