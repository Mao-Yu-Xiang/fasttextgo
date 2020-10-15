#pragma once

#ifdef __cplusplus
extern "C" {
#endif
void load_model(char *name, char *path);
void remove_model(char *name);
int predict(char* name, char *query, float *prob, char **buf, int *count, int k, int buf_sz);
int predictMaxIntention(char* name, char *query, float *prob, char **buf, int *count, int buf_sz);
int getVector(char *name, char *word, float *vector);
int getDimension(char *name);
int getSimilar(char *name, char *query, int k, char **words, float *scores, int *count, int buf_sz);
int getWordN(char *name);
int getWords(char *name, char **words, int buf_sz);
#ifdef __cplusplus
}
#endif
