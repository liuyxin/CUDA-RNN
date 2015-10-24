#include "hardware.h"
void getDevicesinfo() {
	Devices::instance()->print_devname();
	Devices::instance()->max_ThreadsPerBlock();
	Devices::instance()->max_ThreadsDim();
	Devices::instance()->max_GridDims();
	Devices::instance()->get_sharedmemorysize();
}
