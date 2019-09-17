#pragma once

#ifdef __cplusplus
extern "C" {
#endif
void load_model(char *name, char *path);
int predict(char* name, char *query, float *prob, char **buf, int *count, int k, int buf_sz);
int predictMaxIntention(char* name, char *query, float *prob, char **buf, int *count, int buf_sz);
int getVector(char *name, char *word, float *vector);
int getDimension(char *name);

#ifdef __cplusplus
}
#endif
