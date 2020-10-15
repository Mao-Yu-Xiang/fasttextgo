#include <iostream>
#include <istream>
#include <sstream>
#include <string>
#include "fasttext_wrapper.h"
#include "fasttext.h"
#include "real.h"
#include <streambuf>
#include <cstring>
#include <map>

extern "C" {

struct membuf : std::streambuf
{
    membuf(char* begin, char* end) {
        this->setg(begin, begin, end);
    }
};

std::map<std::string, std::shared_ptr<fasttext::FastText>> g_fasttext_model;

void load_model(char *name, char *path) {
	std::shared_ptr<fasttext::FastText> model = std::make_shared<fasttext::FastText>();
	model->loadModel(std::string(path));
	g_fasttext_model[std::string(name)]=model;
}

void remove_model(char *name) {
  if (g_fasttext_model.find(std::string(name)) != g_fasttext_model.end()) {
    g_fasttext_model.erase(std::string(name));
  }
}

//get top k result
int predict(char* name, char *query, float *prob, char **buf, int *count, int k, int buf_sz) {
  membuf sbuf(query, query + strlen(query));
  std::istream in(&sbuf);

  std::vector<std::pair<fasttext::real, std::string>> predictions;
  try {
		  g_fasttext_model.at(std::string(name))->predictLine(in, predictions, k, 0.0);
		
		  int i=0;
		  for (auto it = predictions.cbegin(); it != predictions.cend() && i<k; it++) {
		    *(prob+i) = (float) it->first;
		    strncpy(*(buf+i), it->second.c_str(), buf_sz);
			i++;
		  }
		  *count=i;
		  return 0;
  } catch (const std::exception& e) { 
		return 1;
  }
}

int predictMaxIntention(char* name, char *query, float *prob, char **buf, int *count, int buf_sz) {
  membuf sbuf(query, query + strlen(query));
  std::istream in(&sbuf);

  std::vector<std::pair<fasttext::real, std::string>> predictions;
  try {
		  g_fasttext_model.at(std::string(name))->predictLineMaxIntention(in, predictions);

		  int i=0;
		  for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
		    *(prob+i) = (float) it->first;
		    strncpy(*(buf+i), it->second.c_str(), buf_sz);
			i++;
		  }
		  *count=i;
		  return 0;
  } catch (const std::exception& e) {
		return 1;
  }
}
}

int getVector(char *name, char *word, float *vector) {
  int dim = g_fasttext_model.at(std::string(name))->getDimension();

  fasttext::Vector out(dim);
  try {
    g_fasttext_model.at(std::string(name))->getWordVector(out, std::string(word));
    for (int i=0;i < dim; i++) {
      *(vector+i) = (float) out[i];
    }
    return 0;
  } catch (const std::exception& e) {
    return 1;
  }
}

int getDimension(char *name) {
  return g_fasttext_model.at(std::string(name))->getDimension();
}

int getWordN(char *name) {
  std::shared_ptr<const fasttext::Dictionary> dict = g_fasttext_model.at(std::string(name))->getDictionary();
  return (dict->nwords)();
}

int getSimilar(char *name, char *query, int k, char **words, float *scores, int *count, int buf_sz) {
  try {
    std::vector<std::pair<fasttext::real, std::string>> neighbors = g_fasttext_model.at(std::string(name))->getNN(query, k);

    int i = 0;
    for (auto it = neighbors.cbegin(); it != neighbors.cend(); it++) {
      *(scores+i) = (float) it->first;
      strncpy(*(words+i), it->second.c_str(), buf_sz);
      i++;
    }
    *count=i;
    return 0;
  } catch (const std::exception& e) {
    return 1;
  }
}

int getWords(char *name, char **words, int buf_sz) {
  try {
    std::shared_ptr<const fasttext::Dictionary> dict = g_fasttext_model.at(std::string(name))->getDictionary();
    for (int i= 0; i < (dict->nwords)(); i++) {
      strncpy(*(words+i), (dict->getWord)(i).c_str(), buf_sz);
    }
    return 0;
  } catch (const std::exception& e) {
    return 1;
  }
}
