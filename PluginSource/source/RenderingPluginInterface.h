// RenderingPluginInterface.h
#pragma once

#include <npp.h>

struct BGRDataItem {
	Npp8u* data;
	int width;
	int height;
	int stride;
	int idx;
};

void QueueBGRData(Npp8u* bgrData, int width, int height, int stride, int idx);