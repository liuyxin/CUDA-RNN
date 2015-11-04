#ifndef READDATA_H
#define READDATA_H
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include "Config.h"
using namespace std;
struct singleWord{
	std::string word;
    int label;
    singleWord(string a, int b) : word(a), label(b){}
};
void readdata(string path, vector<vector<singleWord> > &traindata,
		vector<vector<singleWord> > &testdata, vector<string> &re_label,
		map<string, int> &labelmap);
bool isNumber(string &str);
void removeNumber(vector<vector<singleWord> >& data);
void getWordMap(const vector<vector<singleWord> >& data,
		map<string, int> &map, vector<string> &re_map);
void
resolutioner(const std::vector<std::vector<singleWord> > &data,
		std::vector<std::vector<int> > &resol,
		std::vector<std::vector<int> > &labels,
		std::map<string, int> &wordmap);
#endif
