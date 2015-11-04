#include "readdata.h"

using namespace std;

void readdata(string path, vector<vector<singleWord> >& traindata,
		vector<vector<singleWord> >& testdata, vector<string>& re_labelmap,
		map<string, int>& labelmap) {
	vector<vector<singleWord> > data;
	ifstream infile(path.c_str());
	string line;
	vector<singleWord> sentence;
	labelmap["___PADDING___"] = 0;
	re_labelmap.push_back("___PADDING___");
	int counter = 1;
	while (getline(infile, line)) {
		if (line.empty() || line[0] == ' ') {
			if (!sentence.empty()) {
				data.push_back(sentence);
				sentence.clear();
			}
		} else {
			istringstream iss(line);
			string tmpword;
			string tmplabel;
			iss >> tmpword >> tmplabel;
			if (labelmap.find(tmplabel) == labelmap.end()) {
				labelmap[tmplabel] = counter++;  //new label
				re_labelmap.push_back(tmplabel);
			}
			singleWord tmpsw(tmpword, labelmap[tmplabel]);
			sentence.push_back(tmpsw);
		}
	}
	if (!sentence.empty()) {
		data.push_back(sentence);
		sentence.clear();
	}
//	random_shuffle(data.begin(), data.end());
	Config::instance()->set_traintest_num(data.size());
	for (int i = 0; i < data.size(); ++i) {
		if (i < Config::instance()->get_train_num())
			traindata.push_back(data[i]);
		else
			testdata.push_back(data[i]);
	}
	data.clear();
//	vector<vector<singleWord> >().swap(data);
}

bool
isNumber(string &str){
    if(str.empty()) return false;
    for(int i = 0; i < str.size(); i++){
        if(str[i] < '0' || str[i] > '9') return false;
    }
    return true;
}

void
removeNumber(vector<vector<singleWord> >& data){
    for(int i = 0; i < data.size(); i++){
        for(int j = 0; j < data[i].size(); j++){
            if(isNumber(data[i][j].word)) data[i][j].word = "___DIGIT___";
        }
    }
}

void
getWordMap(const vector<vector<singleWord> >& data, map<string, int> &map, vector<string> &re_map){

//    map.clear();
//    re_map.clear();
    for(int i = 0; i < data.size(); i++){
        for(int j = 0; j < data[i].size(); j++){
            if(map.find(data[i][j].word) == map.end()){
                map[data[i][j].word] = re_map.size();
                re_map.push_back(data[i][j].word);
            }
        }
    }
    map["___UNDEFINED___"] = re_map.size();
    re_map.push_back("___UNDEFINED___");
    map["___PADDING___"] = re_map.size();
    re_map.push_back("___PADDING___");
}

void resolutioner(const std::vector<std::vector<singleWord> > &data,
		std::vector<std::vector<int> > &resol,
		std::vector<std::vector<int> > &labels,
		std::map<string, int> &wordmap){

    std::vector<singleWord> tmpvec;
    std::vector<int> tmpresol;
    std::vector<int> tmplabel;
    for(int i = 0; i < data.size(); i++){
        tmpvec.clear();
        tmpvec = data[i];
        singleWord tmpsw("___PADDING___", 0);
        int nGram = Config::instance()->get_ngram();
        int len = (int)(nGram / 2);
        for(int j = 0; j < len; j++){
            tmpvec.insert(tmpvec.begin(), tmpsw);
            tmpvec.push_back(tmpsw);
        }

//        printf("\ntmpvec.size():%d,nGram:%d\n",tmpvec.size(),nGram);
        for(int j = 0; j < tmpvec.size() -nGram + 1; j++){
            tmpresol.clear();
            tmplabel.clear();
            for(int k = 0; k < nGram; k++){
                if(wordmap.find(tmpvec[j + k].word) == wordmap.end()){
                    tmpresol.push_back(wordmap["___UNDEFINED___"]);
                }else{
                    tmpresol.push_back(wordmap[tmpvec[j + k].word]);
                }
                tmplabel.push_back(tmpvec[j + k].label);
            }
            resol.push_back(tmpresol);
            labels.push_back(tmplabel);
        }
    }
}
