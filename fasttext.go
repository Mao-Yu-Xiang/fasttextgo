package fasttextgo

// #cgo LDFLAGS: -L${SRCDIR} -lfasttext -lstdc++ -lm
// #include <stdlib.h>
// void load_model(char *name, char *pathZ);
// int predict(char* name, char *query, float *prob, char **buf, int *count, int k, int buf_sz);
// int predictMaxIntention(char* name, char *query, float *prob, char **buf, int *count, int buf_sz);
// int getVector(char *name, char *word, float *vector);
// int getDimension(char *name);
// int getSimilar(char *name, char *query, int k, char **words, float *scores, int *count, int buf_sz);
// int getWordN(char *name);
// int getWords(char *name, char **words, int buf_sz);
import "C"
import (
	"errors"
	"unsafe"
)

// LoadModel - load FastText model
func LoadModel(name, path string) {
	p1 := C.CString(name)
	p2 := C.CString(path)

	C.load_model(p1, p2)

	C.free(unsafe.Pointer(p1))
	C.free(unsafe.Pointer(p2))
}

// Predict - predict, return the topN predicted label and their corresponding probability
func Predict(name, sentence string, topN int) (map[string]float32, error) {
	result := make(map[string]float32)

	//add new line to sentence, due to the fasttext assumption
	sentence += "\n"

	cprob := make([]C.float, topN, topN)
	buf := make([]*C.char, topN, topN)
	var resultCnt C.int
	for i := 0; i < topN; i++ {
		buf[i] = (*C.char)(C.calloc(64, 1))
	}

	np := C.CString(name)
	data := C.CString(sentence)

	ret := C.predict(np, data, &cprob[0], &buf[0], &resultCnt, C.int(topN), 64)
	if ret != 0 {
		return result, errors.New("error in prediction")
	} else {
		for i := 0; i < int(resultCnt); i++ {
			result[C.GoString(buf[i])] = float32(cprob[i])
		}
	}
	//free the memory used by C
	C.free(unsafe.Pointer(data))
	C.free(unsafe.Pointer(np))
	for i := 0; i < topN; i++ {
		C.free(unsafe.Pointer(buf[i]))
	}

	return result, nil
}

func PredictMaxIntention(name, sentence string) ([]string, []float32, error) {
	resultLabel := make([]string, 0, 6)
	resultScore := make([]float32, 0, 6)

	//add new line to sentence, due to the fasttext assumption
	sentence += "\n"

	cprob := make([]C.float, 6, 6)
	buf := make([]*C.char, 6, 6)
	var resultCnt C.int
	for i := 0; i < 6; i++ {
		buf[i] = (*C.char)(C.calloc(64, 1))
	}

	np := C.CString(name)
	data := C.CString(sentence)

	ret := C.predictMaxIntention(np, data, &cprob[0], &buf[0], &resultCnt, 64)
	if ret != 0 {
		return resultLabel, resultScore, errors.New("error in prediction")
	} else {
		for i := 0; i < int(resultCnt); i++ {
			resultLabel = append(resultLabel, C.GoString(buf[i]))
			resultScore = append(resultScore, float32(cprob[i]))
		}
	}
	//free the memory used by C
	C.free(unsafe.Pointer(data))
	C.free(unsafe.Pointer(np))
	for i := 0; i < 6; i++ {
		C.free(unsafe.Pointer(buf[i]))
	}

	return resultLabel, resultScore, nil
}

func GetWordVector(name, word string) ([]float32, error) {
	n := C.CString(name)
	w := C.CString(word)

	dim := int(C.getDimension(n))
	resultVector := make([]float32, dim, dim)
	vector := make([]C.float, dim, dim)
	ret := C.getVector(n, w, &vector[0])
	if ret != 0 {
		return resultVector, errors.New("error in word2vector")
	} else {
		for i := 0; i < dim; i++ {
			resultVector[i] = float32(vector[i])
		}
	}
	//free the memory used by C
	C.free(unsafe.Pointer(w))
	C.free(unsafe.Pointer(n))
	// C.free(unsafe.Pointer(&vector))

	return resultVector, nil
}

func GetWordVector64(name, word string) ([]float64, error) {
	n := C.CString(name)
	w := C.CString(word)

	dim := int(C.getDimension(n))
	resultVector := make([]float64, dim, dim)
	vector := make([]C.float, dim, dim)
	ret := C.getVector(n, w, &vector[0])
	if ret != 0 {
		return resultVector, errors.New("error in word2vector")
	} else {
		for i := 0; i < dim; i++ {
			resultVector[i] = float64(vector[i])
		}
	}
	//free the memory used by C
	C.free(unsafe.Pointer(w))
	C.free(unsafe.Pointer(n))
	// C.free(unsafe.Pointer(&vector))

	return resultVector, nil
}

func GetMostSimilar(name, query string, top int) (map[string]float32, error) {
	n := C.CString(name)
	q := C.CString(query)

	scores := make([]C.float, top, top)
	words := make([]*C.char, top, top)

	var resultCnt C.int
	for i := 0; i < top; i++ {
		words[i] = (*C.char)(C.calloc(128, 1))
	}

	result := make(map[string]float32, 0)

	ret := C.getSimilar(n, q, C.int(top), &words[0], &scores[0], &resultCnt, 128)
	if ret != 0 {
		return result, errors.New("error in word2vector")
	} else {
		for i := 0; i < int(resultCnt); i++ {
			result[C.GoString(words[i])] = float32(scores[i])
		}
	}
	//free the memory used by C
	C.free(unsafe.Pointer(n))
	C.free(unsafe.Pointer(q))
	for i := 0; i < top; i++ {
		C.free(unsafe.Pointer(words[i]))
	}

	return result, nil
}

func GetWordN(name string) int {
	n := C.CString(name)
	//free the memory used by C
	C.free(unsafe.Pointer(n))
	return int(C.getWordN(n))
}

func GetDimension(name string) int {
	n := C.CString(name)
	dim := int(C.getDimension(n))
	C.free(unsafe.Pointer(n))
	return dim
}

func GetWords(name string) ([]string, error) {
	n := C.CString(name)
	number := int(C.getWordN(n))

	words := make([]*C.char, number, number)

	for i := 0; i < number; i++ {
		words[i] = (*C.char)(C.calloc(128, 1))
	}

	ret := C.getWords(n, &words[0], 128)
	result := make([]string, number, number)
	if ret != 0 {
		return result, errors.New("error in word2vector")
	} else {
		for i := 0; i < number; i++ {
			result[i] = C.GoString(words[i])
		}
	}

	//free the memory used by C
	C.free(unsafe.Pointer(n))
	for i := 0; i < number; i++ {
		C.free(unsafe.Pointer(words[i]))
	}

	return result, nil
}
