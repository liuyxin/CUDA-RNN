#include "Config.h"

string Config::read_2_string(string File_name) {
	char *pBuf;
	FILE *pFile = NULL;
	char logStr[1025];
	if (!(pFile = fopen(File_name.c_str(), "r"))) {
		sprintf(logStr, "Can not find this file.");
		return 0;
	}
	//move pointer to the end of the file
	fseek(pFile, 0, SEEK_END);
	//Gets the current position of a file pointer.offset
	size_t len = ftell(pFile);
	pBuf = new char[len];
	//Repositions the file pointer to the beginning of a file
	rewind(pFile);
	if (fread(pBuf, 1, len, pFile) != len) {
		printf("fread fail\n");
	}
	fclose(pFile);
	string res = pBuf;
	return res;
}

void Config::deleteSpace() {
	if (m_configStr.empty())
		return;
	size_t pos1, pos2, e, t, n;
	while (1) {
		e = m_configStr.find(' ');
		t = m_configStr.find('\t');
		n = m_configStr.find('\n');
		if (e == std::string::npos && n == std::string::npos
				&& t == std::string::npos)
			break;
		if (e < t || t == std::string::npos)
			pos1 = e;
		else
			pos1 = t;
		if (n < pos1 || pos1 == std::string::npos)
			pos1 = n;
		for (pos2 = pos1 + 1; pos2 < m_configStr.size(); pos2++) {
			if (!(m_configStr[pos2] == '\t' || m_configStr[pos2] == '\n'
					|| m_configStr[pos2] == ' '))
				break;
		}
		m_configStr.erase(pos1, pos2 - pos1);
	}
}

void Config::deleteComment() {
	size_t pos1, pos2;
	while (1) {
		pos1 = m_configStr.find("/");
		if (pos1 == std::string::npos)
			break;
		for (pos2 = pos1 + 1; pos2 < m_configStr.size(); pos2++) {
			if (m_configStr[pos2] == '/')
				break;
		}
		m_configStr.erase(pos1, pos2 - pos1 + 1);
	}
}

bool Config::get_word_bool(string &str, string name) {
	size_t pos = str.find(name + "=");
	int i = pos + 1;
	bool res = true;
	while (1) {
		if (i == str.length())
			break;
		if (str[i] == ';')
			break;
		++i;
	}
	string sub = str.substr(pos, i - pos + 1);
	if (sub[sub.length() - 1] == ';') {
		string content = sub.substr(name.length() + 1,
				sub.length() - name.length() - 2);
		if (!content.compare("true"))
			res = true;
		else
			res = false;
	}
	str.erase(pos, i - pos + 1);
	return res;
}

int Config::get_word_int(string &str, string name) {
	size_t pos = str.find(name);
	int i = pos + 1;
	int res = 1;
	while (1) {
		if (i == str.length())
			break;
		if (str[i] == ';')
			break;
		++i;
	}
	string sub = str.substr(pos, i - pos + 1);
	if (sub[sub.length() - 1] == ';') {
		string content = sub.substr(name.length() + 1,
				sub.length() - name.length() - 2);
		res = atoi(content.c_str());
	}
	str.erase(pos, i - pos + 1);
	return res;
}

float Config::get_word_float(string &str, string name) {
	size_t pos = str.find(name + "=");
	int i = pos + 1;
	float res = 0.0f;
	while (1) {
		if (i == str.length())
			break;
		if (str[i] == ';')
			break;
		++i;
	}
	string sub = str.substr(pos, i - pos + 1);
	if (sub[sub.length() - 1] == ';') {
		string content = sub.substr(name.length() + 1,
				sub.length() - name.length() - 2);
		res = atof(content.c_str());
	}
	str.erase(pos, i - pos + 1);
	return res;
}

int Config::get_word_type(string &str, string name) {

	size_t pos = str.find(name);
	int i = pos + 1;
	int res = 0;
	while (1) {
		if (i == str.length())
			break;
		if (str[i] == ';')
			break;
		++i;
	}
	string sub = str.substr(pos, i - pos + 1);
	if (sub[sub.length() - 1] == ';') {
		string content = sub.substr(name.length() + 1,
				sub.length() - name.length() - 2);
		if (!content.compare("NL_SIGMOID"))
			res = 0;
		else if (!content.compare("NL_TANH"))
			res = 1;
		else if (!content.compare("NL_RELU"))
			res = 2;
		else if (!content.compare("HIDDEN"))
			res = 0;
		else if (!content.compare("SOFTMAX"))
			res = 1;
	}
	str.erase(pos, i - pos + 1);
	return res;
}

void Config::get_layers_config(string &str,
		SoftMax &SMR) {
	std::vector<string> layers;
	if (str.empty())
		return;
	int head = 0;
	int tail = 0;
	while (1) {
		if (head == str.length())
			break;
		if (str[head] == '$') {
			tail = head + 1;
			while (1) {
				if (tail == str.length())
					break;
				if (str[tail] == '&')
					break;
				++tail;
			}
			string sub = str.substr(head, tail - head + 1);
			if (sub[sub.length() - 1] == '&') {
				sub.erase(sub.begin() + sub.length() - 1);
				sub.erase(sub.begin());
				layers.push_back(sub);
			}
			str.erase(head, tail - head + 1);
		} else
			++head;
	}
	for (int i = 0; i < layers.size(); i++) {
		int type = get_word_type(layers[i], "LAYER");
		switch (type) {
		case 0: {
			int nn = get_word_int(layers[i], "NUM_HIDDEN_NEURONS");
			double wd = get_word_float(layers[i], "WEIGHT_DECAY");
			double dr = get_word_float(layers[i], "DROPOUT_RATE");
			HiddenConfigs.push_back(HiddenConfig(nn, wd, dr));
			break;
		}
		case 1: {
			int classnum = get_word_int(layers[i], "NUM_CLASSES");
			double wd = get_word_float(layers[i], "WEIGHT_DECAY");
			SMR.set_NumClasses(classnum);
			SMR.set_WeightDecay(wd);
			break;
		}
		}
	}
}

void Config::init(string path,
		SoftMax &SMR) {
	m_configStr = read_2_string(path);
	deleteComment();
	deleteSpace();
	get_layers_config(m_configStr,SMR);
	use_log = get_word_bool(m_configStr, "USE_LOG");
	batch_size = get_word_int(m_configStr, "BATCH_SIZE");
	non_linearity = get_word_type(m_configStr, "NON_LINEARITY");
	training_epochs = get_word_int(m_configStr, "TRAINING_EPOCHS");
	lrate_w = get_word_float(m_configStr, "LRATE_W");
	lrate_b = get_word_float(m_configStr, "LRATE_B");
	iter_per_epo = get_word_int(m_configStr, "ITER_PER_EPO");
	ngram = get_word_int(m_configStr, "NGRAM");
	training_percent = get_word_float(m_configStr, "TRAINING_PERCENT");
	cout
			<< "****************************************************************************"
			<< endl
			<< "**                    READ CONFIG FILE COMPLETE                             "
			<< endl
			<< "****************************************************************************"
			<< endl << endl;

	for (int i = 0; i < HiddenConfigs.size(); i++) {
		cout << "***** hidden layer: " << i << " *****" << endl;
		cout << "NumHiddenNeurons = " << HiddenConfigs[i].get_NeuronNum()
				<< endl;
		cout << "WeightDecay = " << HiddenConfigs[i].get_WeightDecay() << endl;
		cout << "DropoutRate = " << HiddenConfigs[i].get_DropoutRate() << endl << endl;
	}
	cout << "***** softmax layer: *****" << endl;
	//    cout<<"NumClasses = "<<softmaxConfig.NumClasses<<endl;
	cout << "WeightDecay = " << SMR.get_WeightDecay() << endl << endl;
	cout << "***** general config *****" << endl;
	cout << "use_log = " << use_log << endl;
	cout << "batch size = " << batch_size << endl;
	cout << "non-linearity method = " << non_linearity << endl;
	cout << "training epochs = " << training_epochs << endl;
	cout << "learning rate for weight matrices = " << lrate_w << endl;
	cout << "learning rate for bias = " << lrate_b << endl;
	cout << "iteration per epoch = " << iter_per_epo << endl;
	cout << "ngram = " << ngram << endl;
	cout << "training percent = " << training_percent << endl;
	cout << endl;
}
