#pragma once
#include<opencv2/opencv.hpp>
#include<string>
#include <numeric>
#define PI 3.14
#define LOGOTYPE CV_32FC1

enum CutLocation {
	UPLEFT,
	UPRIGHT,
	DOWNLEFT,
	DOWNRIGHT,
	CENTER
};

int dwt(cv::Mat& img) {
	int width = img.cols - img.cols % 2;
	int height = img.rows - img.rows % 2;
	//if (width % 2 || height % 2) return -1;
	cv::Mat righttmp(height, width / 2, CV_32FC1);
	cv::Mat downtmp(height / 2, width, CV_32FC1);
	img.colRange(width / 2, width).copyTo(righttmp);
	for (int i = 0; i < height; ++i) {
		float* p = img.ptr<float>(i);
		float* q = img.ptr<float>(i);
		float* q1 = img.ptr<float>(i);
		++q1;
		float* tmpp = righttmp.ptr<float>(i);
		for (int j = 0; j < width; j += 2) {
			*p = (*q + *q1)/2;
			*tmpp = *p - *q1;
			++p; q += 2; q1 += 2; ++tmpp;
			//img.at<float>(i, j / 2) = (img.at<float>(i, j) + img.at<float>(i, j + 1)) / 2;
			//righttmp.at<float>(i, j / 2) = img.at<float>(i, j / 2) - img.at<float>(i, j + 1);
		}
	}
	righttmp.copyTo(img.colRange(width / 2, width));
	img.rowRange(height / 2, height).copyTo(downtmp);
	for (int j = 0; j < width; ++j) {
		for (int i = 0; i < height; i += 2) {
			img.at<float>(i / 2, j) = (img.at<float>(i, j) + img.at<float>(i + 1, j)) / 2;
			downtmp.at<float>(i / 2, j) = img.at<float>(i / 2, j) - img.at<float>(i + 1, j);
		}
	}
	downtmp.copyTo(img.rowRange(height / 2, height));
	//cv::imshow("dwt", img/256); cv::waitKey(0);
	return 0;
}

int idwt(cv::Mat& img) {
	int width = img.cols - img.cols % 2;
	int height = img.rows - img.rows % 2;
	//if (width % 2 || height % 2) return -1;
	cv::Mat lefttmp(height, width / 2, CV_32FC1);
	cv::Mat uptmp(height / 2, width, CV_32FC1);
	//cv::imshow("to be idwt", img / 256); cv::waitKey(0);
	img.colRange(0, width / 2).copyTo(lefttmp);
	for (int i = 0; i < height; ++i) {
		float* p = img.ptr<float>(i);
		float* q = img.ptr<float>(i);
		q += width / 2;
		float* tmpp = lefttmp.ptr<float>(i);
		for (int j = 0; j < width; j += 2) {
			*p = *tmpp + *q;
			*++p = *tmpp - *q;
			++p; ++tmpp; ++q;
			//img.at<float>(i, j) = lefttmp.at<float>(i, j / 2) + img.at<float>(i, j / 2 + width / 2);
			//img.at<float>(i, j + 1) = lefttmp.at<float>(i, j / 2) - img.at<float>(i, j / 2 + width / 2);
		}
	}
	img.rowRange(0, height / 2).copyTo(uptmp);
	for (int j = 0; j < width; ++j) {
		for (int i = 0; i < height; i += 2) {
			img.at<float>(i, j) = uptmp.at<float>(i / 2, j) + img.at<float>(i / 2 + height / 2, j);
			img.at<float>(i + 1, j) = uptmp.at<float>(i / 2, j) - img.at<float>(i / 2 + height / 2, j);
		}
	}
	//cv::imshow("idwt", img / 256); cv::waitKey(0);
	return 0;
}

double getmse(const cv::Mat& img1, const cv::Mat& img2) {
	if (img1.cols != img2.cols || img1.rows != img2.rows) return -1;
	cv::Mat abs(img1.rows,img1.cols,CV_32F);
	cv::absdiff(img1, img2, abs);
	abs = abs.mul(abs);
	auto s = cv::sum(abs);
	return (double)s[0] / img1.total();
}

double getpsnr(const cv::Mat& img1, const cv::Mat& img2) {
	double mse = getmse(img1, img2);
	return 20.0*log10(255 / (mse+0.0000000001));
}

double getncg(const cv::Mat& img) {
	if (img.channels() != 1) return -1;
	cv::Mat Mx, Mx_t, My;
	cv::reduce(img, Mx, 1, cv::REDUCE_AVG, CV_32FC1);
	cv::reduce(img, My, 0, cv::REDUCE_AVG, CV_32FC1);
	Mx_t = Mx.t();
	auto V = [](cv::Mat& m, std::function<float(float)> fun) {
		float* ptr = m.ptr<float>(0);
		int len = m.cols;
		double v = 0;
		for (int i = 0; i < len; ++i) {
			v += *ptr++*fun(PI / (double)len + (i * 2 * PI / (double)len));
		}
		return v;
	};
	double vx1, vx2, vy1, vy2;
	vx1 = V(Mx_t, static_cast<float(*)(float)>(&cos));
	vx2 = V(Mx_t, static_cast<float(*)(float)>(&sin));
	vy1 = V(My, static_cast<float(*)(float)>(&cos));
	vy2 = V(My, static_cast<float(*)(float)>(&sin));
	return sqrt(
		vx1*vx1 + vx2 * vx2 + vy1 * vy1 + vy2 * vy2
	);
}

double getnc(const cv::Mat& img, const cv::Mat& ori, bool binary = false) {
	if (img.cols != ori.cols || img.rows != ori.rows) return -1;
	if (binary) {
		if (img.channels() != 1 || ori.channels() != 1) return -1;
		cv::Mat img_ = img > 0;
		cv::Mat ori_ = ori > 0;
		return (double)cv::sum(img_.dot(ori_))[0] /
			sqrt(cv::sum(img_.dot(img_))[0]) / sqrt(cv::sum(ori_.dot(ori_))[0]);
	}
	else {
		//cv::Mat img_, ori_;
		
		if (img.channels() != ori.channels()) return -1;

		//cv::normalize(img, img_, 0, 1, cv::NORM_MINMAX, CV_MAKETYPE(CV_32F, img.channels()));
		//cv::normalize(ori, ori_, 0, 1, cv::NORM_MINMAX, CV_MAKETYPE(CV_32F, img.channels()));

		int channels = img.channels();
		auto ab = cv::sum(img.dot(ori));
		auto aa = cv::sum(img.dot(img));
		auto bb = cv::sum(ori.dot(ori));
		double sum_ab = 0;
		double sum_aa = 0;
		double sum_bb = 0;
		for (int i = 0; i < channels; ++i) {
			sum_ab += ab[i];
			sum_aa += aa[i];
			sum_bb += bb[i];
		}
		return sum_ab / sqrt(sum_aa) / sqrt(sum_bb);
	}
}

int embed(cv::Mat &img, cv::Mat & mark, double scale)
{
	cv::dct(img, img);
	int width = img.cols;
	int height = img.rows;
	const int blocksize = 16;
	
	cv::Mat w;
	cv::Mat u;
	cv::Mat vt;
	int wi = 0;
	int wj = 0;
	int midx = 0;
	int mlen = mark.rows;


	/*
	for (int bi = 0; bi < height / blocksize && midx<mlen; ++bi) {
		for (int bj = 0; bj < width / blocksize && midx<mlen; ++bj) {
			cv::SVD::compute(
				img(cv::Range(bi*blocksize, (bi + 1)*blocksize), cv::Range(bj*blocksize, (bj + 1)*blocksize)),
				w,
				u,
				vt
			);
			//std::cout <<w.at<float>(0, 0);
			w.at<float>(0, 0) += scale * mark.at<float>(midx++, 0);
			//std::cout << "\t" << w.at<float>(0, 0) << "\n";
			cv::Mat back = u * cv::Mat::diag(w) * vt;
			back.copyTo(
				img(cv::Range(bi*blocksize, (bi + 1)*blocksize), cv::Range(bj*blocksize, (bj + 1)*blocksize))
			);
		}
	}
	//std::cout << "\n\n\n";
	*/
	cv::SVD::compute(img, w, u, vt,cv::SVD::FULL_UV);
	int maxi = w.rows > mark.rows ? mark.rows : w.rows;
	for (int i = 0; i < maxi; ++i) {
		//std::cout << w.at<float>(i, 0)<<"\n";
		w.at<float>(i, 0) += scale * mark.at<float>(i, 0);
		
	}
	//std::cout << "\n\n\n";
	cv::Mat diag_w(img.rows, img.cols, img.type(), cv::Scalar::all(0));
	cv::Mat::diag(w).copyTo(diag_w(cv::Range(0, w.rows), cv::Range(0, w.rows)));
	cv::Mat new_img = u * diag_w*vt;
	cv::dct(new_img, img, cv::DCT_INVERSE);
	return 0;
}

int extract(cv::Mat & img, cv::Mat & origin, cv::Mat & mark, double scale)
{
	const int blocksize = 16;
	int width = img.cols;
	int height = img.rows;
	if (origin.cols != width || origin.rows != height) return -1;
	//cv::imshow("img", img/256); cv::waitKey(0);
	//cv::imshow("origin", origin/256); cv::waitKey(0);
	mark.setTo(cv::Scalar::all(0));
	cv::Mat w;
	cv::Mat u;
	cv::Mat vt;
	cv::Mat originw;
	int wi = 0;
	int wj = 0;
	int midx = 0;
	int mlen = mark.rows;
	cv::dct(img, img);
	cv::dct(origin, origin);

	/*
	for (int bi = 0; bi < height / blocksize && midx < mlen; ++bi) {
		for (int bj = 0; bj < width / blocksize && midx < mlen; ++bj) {
			cv::SVD::compute(
				img(cv::Range(bi*blocksize, (bi + 1)*blocksize), cv::Range(bj*blocksize, (bj + 1)*blocksize)),
				w
			);
			cv::SVD::compute(
				origin(cv::Range(bi*blocksize, (bi + 1)*blocksize), cv::Range(bj*blocksize, (bj + 1)*blocksize)),
				originw
			);
			mark.at<float>(midx, 0) = (w.at<float>(0, 0) - originw.at<float>(0, 0)) / scale;
			//std::cout << w.at<float>(0, 0) << "\t" << originw.at<float>(0, 0)<<"\n";
			//if (midx != 0 && mark.at<float>(midx, 0) > mark.at<float>(midx - 1, 0)) {	
				//mark.at<float>(midx, 0) = mark.at<float>(midx - 1, 0);}
			++midx;
		}
	}
	//std::cout << "\n\n\n";
	*/
	cv::Mat origin_w;
	cv::SVD::compute(img, w);
	cv::SVD::compute(origin, origin_w);
	int maxi = w.rows > mark.rows ? mark.rows : w.rows;
	bool reverse;
	for (int i = 0; i < maxi; ++i) {
		mark.at<float>(i, 0) = (w.at<float>(i, 0) - origin_w.at<float>(i, 0)) / scale;
		if (i == 0) reverse = mark.at<float>(i, 0) < 0;
		else {
			if (reverse && mark.at<float>(i, 0) > 0) mark.at<float>(i, 0) = 0;
			else if(!reverse && mark.at<float>(i, 0) < 0)mark.at<float>(i, 0) = 0;
		}
		if(reverse) mark.at<float>(i, 0) = -mark.at<float>(i, 0);
	}
	return 0;
}

int Embed(cv::Mat& img, cv::Mat& mark, const std::string& blocks, double scale) {
	if (img.channels() != 1 || mark.channels() != 1) return -1;
	if (img.type() != CV_32FC1) img.convertTo(img, CV_32FC1);
	//cv::namedWindow("will be embeded", 1); cv::imshow("will be embeded", img/256); cv::waitKey(0);
	int width = img.cols;
	int height = img.rows;
	int level = blocks.length();
	if (level == 0) {
		embed(img, mark, scale);
		return 0;
	}
	else if (level < 0) return -4;
	char block = toupper(blocks[0]);
	if (block != 'L'&&block != 'V'&&block != 'H'&&block != 'D') return -3;
	cv::Mat tmp = cv::Mat::zeros(cv::Size(width / 2, height / 2), CV_32FC1);
	dwt(img);
	int offset_x, offset_y = 0;
	switch (block) {
	case 'H':
		offset_x = width / 2;
		break;
	case 'V':
		offset_y = height / 2;
		break;
	case 'D':
		offset_x = width / 2;
		offset_y = height / 2;
		break;
	default:
		break;
	}
	img(cv::Range(offset_y, offset_y + height / 2), cv::Range(offset_x, offset_x + width / 2)).
		copyTo(tmp);
	std::string nextblocks;
	nextblocks.assign(blocks, 1, level - 1);
	Embed(tmp, mark, nextblocks, scale);
	tmp.copyTo(img(cv::Range(offset_y, offset_y + height / 2), cv::Range(offset_x, offset_x + width / 2)));
	//cv::imshow("tmp", tmp/256); cv::waitKey(0);
	idwt(img);
	return 0;
}

int Decode(cv::Mat& img, cv::Mat& origin, cv::Mat& mark, const std::string& blocks, double scale) {

	if (img.channels() != 1 || mark.channels() != 1) return -1;
	if (img.type() != CV_32FC1) img.convertTo(img, CV_32FC1);
	if (origin.type() != CV_32FC1) origin.convertTo(origin, CV_32FC1);
	int width = img.cols;
	int height = img.rows;
	int level = blocks.length();
	//cv::imshow("decode", img / 256); cv::waitKey(0);
	//cv::imshow("decode_2", origin / 256); cv::waitKey(0);
	if (level == 0) {
		extract(img, origin, mark, scale);
		return 0;
	}
	else if (level < 0) return -4;
	char block = toupper(blocks[0]);
	//std::cout << blocks.length() << "\t"<<blocks<<"\n";
	if (block != 'L' && block != 'V' && block != 'H' && block != 'D') return -3;

	cv::Mat tmp = cv::Mat::zeros(cv::Size(width / 2, height / 2), CV_32FC1);
	cv::Mat tmp_origin = cv::Mat::zeros(cv::Size(width / 2, height / 2), CV_32FC1);
	dwt(img);
	dwt(origin);
	//cv::imshow("dwt_decode", img / 256); cv::waitKey(0);
	//cv::imshow("dwt_decode_2", origin / 256); cv::waitKey(0);
	int offset_x = 0;
	int offset_y = 0;
	switch (block) {
	case 'H':
		offset_x = width / 2;
		break;
	case 'V':
		offset_y = height / 2;
		break;
	case 'D':
		offset_x = width / 2;
		offset_y = height / 2;
		break;
	default:
		break;
	}
	img(cv::Range(offset_y, offset_y + height / 2), cv::Range(offset_x, offset_x + width / 2)).
		copyTo(tmp);
	origin(cv::Range(offset_y, offset_y + height / 2), cv::Range(offset_x, offset_x + width / 2)).
		copyTo(tmp_origin);
	std::string nextblocks;
	nextblocks.assign(blocks, 1, level - 1);
	Decode(tmp, tmp_origin, mark, nextblocks, scale);
	return 0;
}

template <typename T>
int tovector(const cv::Mat& img, std::vector<T>& v);

template <typename T>
int tovector(const cv::Mat& img, std::vector<T>& v) {
	if (img.isContinuous()) {
		v.assign((T*)img.data, (T*)img.data + img.total());
	}
	else {
		for (int i = 0; i < img.rows; ++i) {
			v.insert(v.end(), img.ptr<T>(i), img.ptr<T>(i) + img.cols);
		}
	}
	return 0;
}

int add_gaussian(cv::Mat& img, double stddev) {
	int type = img.type();
	int cvttype = CV_MAKETYPE(CV_32F, img.channels());
	img.convertTo(img, cvttype);
	img /= 255;
	cv::Mat noise = cv::Mat(img.size(), img.type());
	cv::randn(noise, 0, stddev);
	img += noise;
	img *= 255;
	img.convertTo(img, type);
	return 0;
}

int add_impulse(cv::Mat& img, double SNR) {
	/*
	cv::Mat mask(img.rows, img.cols, CV_32F);
	cv::randu(mask, 0, 255);
	int width = img.cols;
	int height = img.rows;
	double pro = width * height*(1 - SNR);
	cv::Mat pepper = mask > 255 * (1 - pro / 2);
	cv::Mat salt = mask < 255 * pro / 2;
	img.setTo(255, pepper);
	img.setTo(0, salt);
	*/
	//std::cout << img.type() << "\n";
	int width = img.cols;
	int height = img.rows;
	std::vector<unsigned> increase(img.total());
	std::iota(increase.begin(), increase.end(), 0);
	std::random_shuffle(increase.begin(), increase.end());
	int count = 0;
	int black = SNR * img.total()/2;
	int white = black;
	cv::Vec3b black_vec(0,0,0);
	cv::Vec3b white_vec(255,255,255);
	int x, y;
	for (auto item : increase) {
		//std::cout << item << "\n";
		x = item % width;
		y = item / width;
		if (count < black) {
			img.at<cv::Vec3b>(y, x) = black_vec;
		}
		else if (count < black + white) {
			img.at<cv::Vec3b>(y, x) = white_vec;
		}
		else break;
		++count;
	}
	return 0;
}

int imgcut(cv::Mat& img, CutLocation location, int width, int height, double percent = 0) {
	int imgwidth = img.cols;
	int imgheight = img.rows;
	int imgtype = img.type();
	if (percent != 0) {
		double scale = sqrt(percent*(double)img.total() / width / height);
		width *= scale;
		height *= scale;
	}
	
	cv::Mat black(width, height, imgtype);
	black.setTo(cv::Scalar::all(0));
	int x_offset = 0;
	int y_offset = 0;
	switch (location) {
	case UPLEFT:
		break;
	case UPRIGHT:
		x_offset = imgwidth - width;
		break;
	case DOWNRIGHT:
		y_offset = imgheight - height;
		break;
	case DOWNLEFT:
		x_offset = imgwidth - width;
		y_offset = imgheight - height;
		break;
	case CENTER:
		x_offset = imgwidth / 2 - width / 2;
		y_offset = imgheight / 2 - height / 2;
		break;
	default:
		break;
	}
	black.copyTo(img(cv::Range(y_offset, y_offset + height), cv::Range(x_offset, x_offset + width)));
	return 0;
}

int imgjpeg(cv::Mat& img, int quality) {
	std::vector<uchar> buffer;
	std::vector<int> paras;
	paras.emplace_back(cv::IMWRITE_JPEG_QUALITY);
	paras.emplace_back(quality);
	cv::imencode(".jpg", img, buffer, paras);
	img = cv::imdecode(buffer, -1);
	return 0;
}