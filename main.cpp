#include "tools.h"
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#define NCG 2500
#define ORIGIN_LOGO_TYPE CV_8UC1

std::vector<unsigned> GetKeyFrames(const std::string& video_path,int windowsize) {
	cv::VideoCapture cap(video_path);
	if (!cap.isOpened()) return std::vector<unsigned>(0);
	int frame_num = cap.get(cv::CAP_PROP_FRAME_COUNT);
	int fidx = 1;
	int maxfidx, startfidx;
	double ncg, maxNCG;
	bool window_extended;
	std::vector<unsigned> selected_frames;
	cv::Mat yuv[3];
	cv::Mat frame;
	while (fidx < frame_num) {
		cap.set(cv::CAP_PROP_POS_FRAMES, fidx);
		startfidx = fidx;
		maxfidx = -1;
		window_extended = false;
		//std::cout << "new window: ";
		for (int i = 0; i < windowsize&&fidx < frame_num; ++i, ++fidx) {
			cap.read(frame);
			cv::cvtColor(frame, frame, cv::COLOR_BGR2YUV);
			cv::split(frame, yuv);
			ncg = getncg(yuv[0]);
			//std::cout << ncg<<" ";
			if (maxfidx == -1 || ncg > maxNCG) {
				maxNCG = ncg;
				maxfidx = fidx;
			}
		}
		if (maxfidx == -1 || maxfidx == frame_num - 1) break;

		//window extend
		
		if (maxfidx == startfidx + windowsize - 1) {
			
			window_extended = true;
			/*
			while (++fidx < frame_num) {
				cap.read(frame);
				cv::cvtColor(frame, frame, cv::COLOR_BGR2YUV);
				cv::split(frame, yuv);
				ncg = getncg(yuv[0]);
				if (ncg <= maxNCG) break;
				maxNCG = ncg;
				maxfidx = fidx;
			}
			if (maxfidx == frame_num - 1) break;
			*/
		}

		selected_frames.emplace_back(maxfidx);
		//std::cout << maxfidx << "\t" << maxNCG << "\n";
		if (window_extended) ++fidx;
		//fidx = maxfidx + 2;
	}
	return selected_frames;
}

int VideoEmbed(
	const std::string& video_path,
	const std::string& wm_path,
	const std::string& o_path,
	const std::string& key_path,
	double scale,
	const std::string& blocks,
	int windowsize
) {
	cv::Mat logo = cv::imread(wm_path, false);
	if (!logo.data) return -1;
	logo.convertTo(logo, CV_32FC1);
	cv::Mat u, w, vt;
	//if (binary) cv::threshold(logo, logo, truncvalue, 1, cv::THRESH_BINARY);
	//cv::normalize(logo, logo, 0, 1, cv::NORM_MINMAX, LOGOTYPE);
	logo = logo / 255;
	//cv::imshow("logo", logo); cv::waitKey(0);
	cv::SVD::compute(logo, w, u, vt, cv::SVD::FULL_UV);
	//std::cout << w << "\n";
	cv::VideoCapture cap(video_path);
	if (!cap.isOpened()) return -1;
	int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	int frame_num = cap.get(cv::CAP_PROP_FRAME_COUNT);
	double fps = cap.get(cv::CAP_PROP_FPS);
	int fourcc = cap.get(cv::CAP_PROP_FOURCC);
	cv::VideoWriter writer(o_path, fourcc, fps, cv::Size(width, height));
	if (!writer.isOpened()) return -2;
	cv::Mat yuv[3];
	cv::Mat frame;
	std::vector<unsigned> selected_frames;
	int fidx = 1;
	int maxfidx, startfidx;
	double ncg, maxNCG;
	bool window_extended;
	//construct selected_frames vector
	while (fidx < frame_num) {
		cap.set(cv::CAP_PROP_POS_FRAMES, fidx); 
		startfidx = fidx;
		maxfidx = -1;
		window_extended = false;
		for (int i = 0; i < windowsize&&fidx < frame_num; ++i, ++fidx) {
			cap.read(frame);
			cv::cvtColor(frame, frame, cv::COLOR_BGR2YUV);
			cv::split(frame, yuv);
			ncg = getncg(yuv[0]);
			//std::cout << ncg<<" ";
			if (maxfidx == -1 || ncg > maxNCG) {
				maxNCG = ncg;
				maxfidx = fidx;
			}
		}
		if (maxfidx == -1||maxfidx==frame_num-1) break;
		
		//window extend
		if (maxfidx == startfidx + windowsize - 1) {
			//std::cout << "window extend\n";
			window_extended = true;
			/*
			while (++fidx < frame_num) {
				cap.read(frame);
				cv::cvtColor(frame, frame, cv::COLOR_BGR2YUV);
				cv::split(frame, yuv);
				ncg = getncg(yuv[0]);
				if (ncg <= maxNCG) break;
				maxNCG = ncg;
				maxfidx = fidx;
			}
			if (maxfidx == frame_num - 1) break;
			*/
		}
		
		selected_frames.emplace_back(maxfidx);
		//std::cout << maxfidx << "\t" << maxNCG << "\n";
		if (window_extended) ++fidx;
		//fidx = maxfidx + 2;
	}
	//for (auto item : selected_frames) std::cout << item << "\n";
	fidx = 0;
	cap.set(cv::CAP_PROP_POS_FRAMES, fidx);
	//embed while writing new video
	std::cout.precision(3);
	std::cout << "embedding...\n";
	for (auto nextselected : selected_frames) {
		while (fidx++ != nextselected) {
			cap >> frame; 
			writer << frame;
			std::cout << (double)fidx / frame_num << "\r";
		}
		cap >> frame;
		cv::cvtColor(frame, frame, cv::COLOR_BGR2YUV);
		cv::split(frame, yuv);
		Embed(yuv[0], w, blocks, scale);
		yuv[0].convertTo(yuv[0], ORIGIN_LOGO_TYPE);
		cv::merge(yuv, 3, frame);
		cv::cvtColor(frame, frame, cv::COLOR_YUV2BGR);
		writer << frame;
	}

	while (fidx++ < frame_num) {
		std::cout << (double)fidx / frame_num << '\r';
		cap >> frame;
		writer << frame;
	}
	std::cout << "\t\t\t\r";
	std::fstream key(key_path, std::ios::out | std::ios::binary);
	if (!key.is_open()) return -1;
	int blockslen = blocks.length();
	std::vector<float> udata;
	std::vector<float> vtdata;
	//std::cout << u << "\n";
	//std::cout << vt << "\n";
	tovector(u, udata);
	tovector(vt, vtdata);
	key.write((const char *)&scale, sizeof(double));
	key.write((const char *)&blockslen, sizeof(int));
	key.write(blocks.data(), blockslen);
	key.write((const char *)&windowsize, sizeof(int));

	key.write((const char *)&logo.rows, sizeof(int));
	key.write((const char *)&logo.cols, sizeof(int));
	//for (auto item : udata) std::cout << item << " ";
	//for (auto item : vtdata) std::cout << item << " ";
	key.write((const char *)udata.data(), udata.size() * sizeof(float));
	key.write((const char *)vtdata.data(), vtdata.size() * sizeof(float));
	
	key.close();
	return 0;
}

int VideoDecode(
	const std::string& video_path, 
	const std::string& key_path, 
	const std::string& o_path,
	bool binary,
	double truncvalue,
	double min_nc,
	int credit_level,
	int minvalid,
	const std::string& origin_path) 
{
	cv::VideoCapture cap(video_path);
	if (!cap.isOpened()) return -1;
	cv::VideoCapture origin_cap;
	if (!origin_path.empty()) {
		origin_cap.open(origin_path);
		if (!origin_cap.isOpened()) return -1;
	}
	int frame_num = cap.get(cv::CAP_PROP_FRAME_COUNT);
	std::fstream key(key_path, std::ios::in | std::ios::binary);
	if (!key.is_open()) return -1;
	double scale;
	int windowsize, blockslen, logo_cols, logo_rows;
	key.read((char *)&scale, sizeof(double));
	key.read((char*)&blockslen, sizeof(int));
	char *tmpblocks = new char[blockslen+1];
	if (!tmpblocks) return -1;
	key.read(tmpblocks, blockslen);
	tmpblocks[blockslen] = '\0';
	std::string blocks = tmpblocks;
	delete tmpblocks;
	key.read((char *)&windowsize, sizeof(int));

	key.read((char *)&logo_rows, sizeof(int));
	key.read((char *)&logo_cols, sizeof(int));
	cv::Mat u(logo_rows, logo_rows, LOGOTYPE);
	cv::Mat vt(logo_cols, logo_cols, LOGOTYPE);
	key.read((char *)u.data, u.total() * sizeof(float));
	key.read((char *)vt.data, vt.total() * sizeof(float));
	//std::cout << u << "\n";
	//std::cout << vt << "\n";
	
	key.close();

	cv::Mat w(logo_cols > logo_rows ? logo_rows : logo_cols, 1, LOGOTYPE);
	cv::Mat yuv[3];
	cv::Mat frame;
	int fidx = 1;
	cap.set(cv::CAP_PROP_POS_FRAMES, fidx);
	int maxfidx, startfidx;
	float ncg, maxNCG;
	bool window_extended;
	struct Selected {
		cv::Mat* mat;
		double sum_nc;
		int no;
		int credit = 0;
	};
	std::vector<Selected*> selectedv;
	cv::Mat final_decoded = cv::Mat::zeros(cv::Size(logo_cols, logo_rows), LOGOTYPE);
	bool isValid = false;
	while (fidx < frame_num) {
		cap.set(cv::CAP_PROP_POS_FRAMES, fidx);
		startfidx = fidx;
		maxfidx = -1;
		window_extended = false;
		for (int i = 0; i < windowsize&&fidx < frame_num; ++i, ++fidx) {
			cap.read(frame);
			cv::cvtColor(frame, frame, cv::COLOR_BGR2YUV);
			cv::split(frame, yuv);
			ncg = getncg(yuv[0]);
			if (maxfidx == -1 || ncg > maxNCG) {
				maxNCG = ncg;
				maxfidx = fidx;
			}
		}
		if (maxfidx == -1 || maxfidx == frame_num - 1) break;
		//window extend
		
		if (maxfidx == startfidx + windowsize - 1) {
			window_extended = true;
			/*
			//std::cout << "window_extend\n";
			while (++fidx < frame_num) {
				cap.read(frame);
				cv::cvtColor(frame, frame, cv::COLOR_BGR2YUV);
				cv::split(frame, yuv);
				ncg = getncg(yuv[0]);
				if (ncg <= maxNCG) break;
				maxNCG = ncg;
				maxfidx = fidx;
			}
			if (maxfidx == frame_num - 1) break;
			*/
		}
		
		//std::cout << "====================\n";
		//std::cout << "select key frame: " << maxfidx << "\n";
		
		cv::Mat pre_yuv[3];
		cv::Mat embed_yuv[3];
		//cv::Mat next_yuv[3];
		cv::Mat origin;

		if (!origin_path.empty()) {
			origin_cap.set(cv::CAP_PROP_POS_FRAMES, maxfidx);
			origin_cap >> frame;
			cv::cvtColor(frame, frame, cv::COLOR_BGR2YUV);
			cv::split(frame, pre_yuv);
			origin = pre_yuv[0];
			cap.set(cv::CAP_PROP_POS_FRAMES, maxfidx);
			cap >> frame;
			cv::cvtColor(frame, frame, cv::COLOR_BGR2YUV);
			cv::split(frame, embed_yuv);
		}
		else {
			cap.set(cv::CAP_PROP_POS_FRAMES, maxfidx - 1);
			cap >> frame;
			cv::cvtColor(frame, frame, cv::COLOR_BGR2YUV);
			cv::split(frame, pre_yuv);
			cap >> frame;
			cv::cvtColor(frame, frame, cv::COLOR_BGR2YUV);
			cv::split(frame, embed_yuv);
			//cap >> frame;
			//cv::cvtColor(frame, frame, cv::COLOR_BGR2YUV);
			//cv::split(frame, next_yuv);
			//if (pre_yuv[0].type() != CV_32FC1) pre_yuv[0].convertTo(pre_yuv[0], CV_32FC1);
			//if (next_yuv[0].type() != CV_32FC1) next_yuv[0].convertTo(next_yuv[0], CV_32FC1);
			origin = pre_yuv[0];
		}

		//cv::imshow("origin", origin); cv::waitKey(0);
		//cv::imshow("embeded", embed_yuv[0]); cv::waitKey(0);
		//cv::GaussianBlur(origin, origin, cv::Size(3, 3), 0, 0.7);
		//cv::GaussianBlur(embed_yuv[0], embed_yuv[0], cv::Size(3, 3), 0, 0.7);
		Decode(embed_yuv[0], origin, w, blocks, scale);
		//std::cout<<w<<"\n";
		cv::Mat diag_w(logo_rows, logo_cols, LOGOTYPE, cv::Scalar::all(0));
		cv::Mat::diag(w).copyTo(diag_w(cv::Range(0, w.rows), cv::Range(0, w.rows)));
		cv::Mat decoded_logo = u * diag_w*vt;
		//std::cout << decoded_logo;
		//cv::normalize(decoded_logo, decoded_logo, 0, 1, cv::NORM_MINMAX, LOGOTYPE);
		//cv::imshow("decoded_logo", decoded_logo); cv::waitKey(0);

		Selected* selected = new Selected();
		selected->mat = new cv::Mat();
		//if (cv::sum(decoded_logo)[0] < 0) *selected->mat = -decoded_logo;
		*selected->mat = decoded_logo;
		selected->no = maxfidx;
		int valid = 0;
		for (auto item:selectedv) {
			double nc = getnc(*item->mat, *selected->mat);
			//std::cout << "NC with " << item->no << ": "<<nc<<"\n";
			if (nc > min_nc) {
				++item->credit;
				++selected->credit;
				item->sum_nc += nc;
				selected->sum_nc += nc;
			}
			if (item->credit >= credit_level) ++valid;
		}
		selectedv.emplace_back(selected);
		if (selected->credit >= credit_level) ++valid;
		if (valid >= minvalid) {
			/*
			valid = true;
			for (auto item : selectedv) {
				if (item->credit >= credit_level) {
					final_decoded += *item->mat;
				}
			}
			final_decoded /= valid;
			*/
			break;
		}
		if(window_extended) ++fidx;
	}
	sort(selectedv.begin(), selectedv.end(), [](Selected*a, Selected*b) {
		return a->sum_nc > b->sum_nc;
	});
	int count = 0;
	int max_sum_nc = -1;
	for (auto item : selectedv) {
		if (max_sum_nc == -1) {
			if (item->credit < 1) break;
			max_sum_nc = item->sum_nc;
		}
		if (item->sum_nc < max_sum_nc*0.95) break;
		final_decoded += *item->mat;
		++count;
	}
	double mean = cv::mean(final_decoded)[0];
	if (binary) cv::threshold(final_decoded, final_decoded, mean, 255, cv::THRESH_BINARY);
	else {
		double min, max;
		cv::minMaxLoc(final_decoded, &min, &max);
		//std::cout << final_decoded << "\n\n";
		//cv::threshold(final_decoded, final_decoded, 1.2, NULL, cv::THRESH_TRUNC);
		//cv::normalize(final_decoded, final_decoded, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		final_decoded = final_decoded * 255;
	}
	cv::imwrite(o_path, final_decoded);
	for (auto item : selectedv) {
		delete item->mat;
	}
	return selectedv.size();
}

int FrameAttack(
	const std::string& video_path,
	const std::string& o_path,
	std::function<void(cv::Mat&)> fun
) {
	cv::VideoCapture cap(video_path);
	if (!cap.isOpened()) return -1;
	int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	double fps = cap.get(cv::CAP_PROP_FPS);
	int frame_num = cap.get(cv::CAP_PROP_FRAME_COUNT);
	int fourcc = cap.get(cv::CAP_PROP_FOURCC);
	cv::VideoWriter writer(o_path, fourcc, fps, cv::Size(width, height));
	if (!writer.isOpened()) return -2;
	int fidx = 0;
	cv::Mat frame;
	//std::cout.precision(3);
	//std::cout << o_path << ":\n";
	while (fidx++ < frame_num) {
		cap >> frame;
		fun(frame);
		writer << frame;
		//std::cout << (double)fidx / frame_num << '\r';
	}
	//std::cout << "\t\t\t\r";
	return 0;
}

std::vector<unsigned> FrameLost(
	const std::string& video_path,
	const std::string& o_path,
	double percent,
	const std::vector<unsigned>& selected_frames
) {
	cv::VideoCapture cap(video_path);
	if (!cap.isOpened()) return{};
	int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	double fps = cap.get(cv::CAP_PROP_FPS);
	int frame_num = cap.get(cv::CAP_PROP_FRAME_COUNT);
	int fourcc = cap.get(cv::CAP_PROP_FOURCC);
	cv::VideoWriter writer(o_path, fourcc, fps, cv::Size(width, height));
	if (!writer.isOpened()) return{};
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<double> dis;
	int fidx = 0;
	cv::Mat frame;
	//std::cout.precision(3);
	//std::cout << o_path << ":\n";
	int count = 0;
	std::vector<unsigned> new_selected_frames;
	while (fidx++ < frame_num) {
		//std::cout << (double)fidx/frame_num << '\r';
		cap >> frame;
		if (dis(mt) > percent) {
			if (std::find(selected_frames.begin(), selected_frames.end(), fidx-1) != selected_frames.end()) {
				new_selected_frames.emplace_back(count);
			}
			writer << frame;
			++count;
		}
	}
	//std::cout << "\t\t\t\r";
	return new_selected_frames;
}

int FrameAvg(
	const std::string& video_path,
	const std::string& o_path,
	int count
) {
	std::cout << video_path << "\n";
	cv::VideoCapture cap(video_path);
	if (!cap.isOpened()) return -1;
	int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	double fps = cap.get(cv::CAP_PROP_FPS);
	int frame_num = cap.get(cv::CAP_PROP_FRAME_COUNT);
	int fourcc = cap.get(cv::CAP_PROP_FOURCC);
	cv::VideoWriter writer(o_path, fourcc, fps, cv::Size(width, height));
	if (!writer.isOpened()) return -1;
	int fidx = 0;
	int i = 0;
	
	using matv = std::vector<cv::Mat>;
	std::function<cv::Mat(matv::iterator, matv::iterator)> sumavgMat;
	sumavgMat = [&sumavgMat](matv::iterator begin, matv::iterator end)->cv::Mat {
		//std::cout << begin->type()<<"\n";
		if (begin + 1 == end) return *begin;
		else return *begin + sumavgMat(begin + 1, end);
	};
	matv frames(count);

	std::cout << o_path << ":\n";
	while (fidx++ < frame_num) {
		cap >> frames[i++];
		if (i == count) {
			i = 0;
			cv::Mat sumframe(height, width, CV_32FC3);
			for (auto& item : frames) item.convertTo(item, CV_32FC3);
			//std::cout << "type:" << frames[0].type() << "\n";
			sumframe = sumavgMat(frames.begin(), frames.end());
			//std::cout << sumframe/count;
			auto avgframe = sumframe / count;
			writer << sumframe/count;
			sumframe.setTo(cv::Scalar::all(0));
			std::cout << (double)fidx / frame_num << '\r';
		}
	}
	if (i != 0) {
		cv::Mat sumframe = sumavgMat(frames.begin(), frames.begin()+i);
		writer << sumframe/i;
	}
	std::cout << "\t\t\r";
	return -1;
}

std::vector<unsigned> VideoSeg(
	const std::string& video_path,
	const std::string& o_path,
	double start,
	double last,
	const std::vector<unsigned>& selected_frames
) {
	if (start >= 1) return {};
	cv::VideoCapture cap(video_path);
	if (!cap.isOpened()) return {};
	int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	double fps = cap.get(cv::CAP_PROP_FPS);
	int frame_num = cap.get(cv::CAP_PROP_FRAME_COUNT);
	int fourcc = cap.get(cv::CAP_PROP_FOURCC);
	cv::VideoWriter writer(o_path, fourcc, fps, cv::Size(width, height));
	if (!writer.isOpened()) return{};
	int start_frame_no = start * frame_num;
	int end_frame_no = (start + last)*frame_num;
	end_frame_no = end_frame_no > frame_num ? frame_num : end_frame_no;
	int fidx = start_frame_no;
	cv::Mat frame;
	cap.set(cv::CAP_PROP_POS_FRAMES, fidx);

	std::vector<unsigned> new_selected_frames;
	for (auto item : selected_frames) {
		if (item < end_frame_no&&item >= start_frame_no)
			new_selected_frames.emplace_back(item - start_frame_no);
	}
	
	//std::cout.precision(3);
	//std::cout << o_path << ":\n";
	int count = 0;
	while (fidx++ < end_frame_no) {
		//std::cout << (double)(fidx-start_frame_no) / (last*frame_num) << '\r';
		cap >> frame;
		writer << frame;
	}
	//std::cout << "\t\t\t\r";
	return new_selected_frames;
}

int test(const std::string& video, const std::string& suffix) {
	const std::string dir = "videosample\\" + video + "\\";
	const std::string wm = "lena64gray.tiff";
	const std::string o_path = dir+video+"_embed"+suffix;
	const std::string key_path = dir+video+"_key.wkey";
	const std::string embeded = dir + video + "_embed" + suffix;
	bool origin_test = false;
	bool regen = true;
	double scale = 10;
	std::string blocks = "HD";
	int windowsize = 10;

	bool binary = false;
	double truncvalue = 0.5;
	double min_nc = 0.8;
	double credit_level = 2;
	double min_valid = 2;
	
	std::vector<unsigned> origin_key_frames;
	std::vector<unsigned> key_frames;

	std::function<void(const std::vector<unsigned>&,const std::vector<unsigned>&,double&,double&)> accuracy =
		[](const std::vector<unsigned>&key_frames,const std::vector<unsigned>& hit,double& acc_rate, double& near_acc_rate) {
		int acc = 0;
		int near_acc = 0;
		for (auto item : hit) {
			auto it = std::find(key_frames.begin(), key_frames.end(), item);
			if (it != key_frames.end()) ++acc;
			else {
				auto pre_it = std::find(key_frames.begin(), key_frames.end(), item-1);
				//auto next_it = std::find(key_frames.begin(), key_frames.end(), item + 1);
				if (pre_it != key_frames.end()) ++near_acc;
			}
		}
		acc_rate = (double)acc / hit.size();
		near_acc_rate = (double)near_acc/hit.size();
		return;
	};

	auto psnr = [&](const std::string&v1, const std::string&v2, bool show = false)->double {
		cv::VideoCapture cap1(v1);
		cv::VideoCapture cap2(v2);
		double sum = 0;
		if (!cap1.isOpened() || !cap2.isOpened()) return 0;
		for (auto fidx : origin_key_frames) {
			cap1.set(cv::CAP_PROP_POS_FRAMES, fidx);
			cap2.set(cv::CAP_PROP_POS_FRAMES, fidx);
			cv::Mat frame1, frame2;
			cap1 >> frame1;
			cap2 >> frame2;
			double psnrvalue = getpsnr(frame1, frame2);
			sum += psnrvalue;
			if(show)std::cout << psnrvalue << "\n";
		}
		return sum / origin_key_frames.size();
	};

	auto decode_and_test = [&](const std::string& o_video) {
		int true_hit = VideoDecode(o_video + suffix, key_path, o_video + ".BMP", binary, truncvalue, min_nc, credit_level, min_valid, {});
		cv::Mat origin = cv::imread(wm, 0);
		cv::Mat decoded = cv::imread(o_video + ".BMP", 0);
		//std::cout << o_video << "\n";
		std::cout << getnc(origin, decoded) << "\t";
		std::cout << getpsnr(origin, decoded) << "\t";
		auto hit = GetKeyFrames(o_video + suffix,windowsize);
		std::cout << key_frames.size()<<"\t";
		std::cout << hit.size() << "\t";
		double acc, near_acc;
		accuracy(key_frames, hit, acc, near_acc);
		std::cout << acc<<"\t"<<near_acc << "\t";
		std::cout << true_hit << "\t";
	
		if (origin_test) {
			VideoDecode(o_video + suffix, key_path, o_video+"_ORIGIN" + ".BMP", binary, truncvalue, min_nc, credit_level, min_valid, dir + video +"_0"+ suffix);
			cv::Mat origin = cv::imread(wm, 0);
			cv::Mat decoded = cv::imread(o_video +"_ORIGIN" ".BMP", 0);
			std::cout << getnc(origin, decoded) << "\t";
			std::cout << getpsnr(origin, decoded) << "\t";
			std::cout << psnr(o_video + suffix, dir + video + suffix);
		}
		std::cout << "\n";
	};

	auto noisg = [&](double stddev) {
		std::ostringstream out;
		out.precision(3);
		out << stddev;
		std::function<void(cv::Mat&)> fun = [&](cv::Mat& img) {add_gaussian(img, stddev); };
		std::string o_video = dir + video + "_noisg_" + out.str();
		if(regen)FrameAttack(embeded, o_video + suffix, fun);
		decode_and_test(o_video);
	};

	auto noisimp = [&](double snr) {
		std::ostringstream out;
		out.precision(3);
		out << snr;
		std::function<void(cv::Mat&)> fun = [&](cv::Mat& img) {add_impulse(img, snr); };
		std::string o_video = dir + video + "_noisimp_" + out.str();
		if (regen)FrameAttack(embeded, o_video + suffix, fun);
		decode_and_test(o_video);
	};

	auto bluravg = [&](int size) {
		std::ostringstream out;
		out.precision(3);
		out << size;
		std::function<void(cv::Mat&)> fun = [&](cv::Mat& img) {cv::blur(img, img, cv::Size(size, size)); };
		std::string o_video = dir + video + "_bluravg_" + out.str();
		if (regen)FrameAttack(embeded, o_video + suffix, fun);
		decode_and_test(o_video);
	};

	auto blurg = [&](int size) {
		std::ostringstream out;
		out.precision(3);
		out << size;
		std::function<void(cv::Mat&)> fun = [&](cv::Mat& img) {cv::GaussianBlur(img, img, cv::Size(size, size), 0, 0.3); };
		std::string o_video = dir + video + "_blurg_" + out.str();
		if (regen)FrameAttack(embeded, o_video + suffix, fun);
		decode_and_test(o_video);
	};

	auto blurmed = [&](int size) {
		std::ostringstream out;
		out.precision(3);
		out << size;
		std::function<void(cv::Mat&)> fun = [&](cv::Mat& img) {cv::medianBlur(img, img, size); };
		std::string o_video = dir + video + "_blurmed_" + out.str();
		if (regen)FrameAttack(embeded, o_video + suffix, fun);
		decode_and_test(o_video);
	};

	auto cut = [&](CutLocation location,double percent) {
		std::ostringstream out;
		out.precision(3);
		out << location;
		out << "_";
		out << percent;
		std::function<void(cv::Mat&)> fun = [&](cv::Mat& img) {imgcut(img, location, 1, 1, percent);};
		std::string o_video = dir + video + "_cut_" + out.str();
		if (regen)FrameAttack(embeded, o_video + suffix, fun);
		decode_and_test(o_video);
	};

	auto jpeg = [&](int quality) {
		std::ostringstream out;
		out.precision(3);
		out << quality;
		std::function<void(cv::Mat&)> fun = [&](cv::Mat& img) {imgjpeg(img, quality); };
		std::string o_video = dir + video + "_jpeg_" + out.str();
		if (regen)FrameAttack(embeded, o_video + suffix, fun);
		decode_and_test(o_video);
	};

	auto lost = [&](double percent) {
		std::ostringstream out;
		out.precision(3);
		out << percent;
		std::string o_video = dir + video + "_lost_" + out.str();
		auto new_key = FrameLost(embeded, o_video+suffix,percent, origin_key_frames);
		key_frames.clear(); key_frames.assign(new_key.begin(), new_key.end());
		decode_and_test(o_video);
	};

	auto seg = [&](double start,double last) {
		std::ostringstream out;
		out.precision(3);
		out << start << "_"<<last;
		std::string o_video = dir + video + "_seg_" + out.str();
		auto new_key = VideoSeg(embeded, o_video + suffix, start,last, origin_key_frames);
		key_frames.clear(); key_frames.assign(new_key.begin(), new_key.end());
		decode_and_test(o_video);
	};

	

	std::cout.setf(std::ios::fixed);
	std::cout.precision(3);
	
	
	
	

	origin_key_frames = GetKeyFrames(dir + video + suffix, windowsize);
	key_frames.assign(origin_key_frames.begin(), origin_key_frames.end());
	
	VideoEmbed(dir + video + suffix, wm, dir + video + "_0"+suffix, key_path, 0, blocks, windowsize);
	VideoEmbed(dir + video + suffix, wm, o_path, key_path, scale, blocks, windowsize);
	origin_test = false;
	decode_and_test(dir + video);

	
	origin_test = true;
	regen = true;

	//for (auto item : key_frames) std::cout << item << "\n";
	psnr(dir + video + suffix, o_path,true);
	decode_and_test(dir + video + "_embed");
	
	//std::cout << "key frame num: " << key_frames.size() << "\n";
	
	noisg(0.003);noisg(0.005);noisg(0.01);noisg(0.05);noisg(0.1);

	noisimp(0.003); noisimp(0.01);  noisimp(0.05); noisimp(0.1); noisimp(0.2);

	bluravg(3);bluravg(5);bluravg(7);

	blurg(3);blurg(5);blurg(7);
	
	blurmed(3);blurmed(5);blurmed(7);

	jpeg(10); jpeg(20); jpeg(30);

	cut(UPLEFT,0.125);cut(CENTER, 0.125); cut(DOWNRIGHT, 0.125); cut(CENTER, 0.2);
	
	regen = true;
	origin_test = false;
	lost(0.1); lost(0.1); lost(0.1); 
	lost(0.2); lost(0.2); lost(0.2);
	lost(0.3); lost(0.3); lost(0.3);
	seg(0.2, 0.3);seg(0.3, 0.5);
	
	
	//VideoSeg(dir + video + suffix, dir + video + "_" + suffix, 0.5, 0.03, {});
	return 0;
}

int maintest() {
	cv::Mat img = cv::imread("lena64gray.tiff",false);
	add_gaussian(img, 0.01);
	cv::imwrite("true_lena64gray.tiff", img);
	cv::Mat img1 = cv::imread("lena64gray.tiff", false);
	cv::Mat img2 = cv::imread("true_lena64gray.tiff", false);
	std::cout << getpsnr(img1, img2);
	return 0;
}

int main(int argc, char* argv[]) {
	//attack("experiment");
	//maintest();
	
	std::cout << "\n\n\t\t\t================view================\n\n";
	test("view", ".mp4");
	std::cout << "\n\n\t\t\t================anim================\n\n";
	test("anim",".mp4");
	std::cout << "\n\n\t\t\t================exp================\n\n";
	test("exp", ".mov");
	std::cout << "\n\n\t\t\t================news================\n\n";
	test("news", ".mp4");
	return 0;
}

//ÎÞ¹¥»÷ÏÂncÖµ=0.99933
