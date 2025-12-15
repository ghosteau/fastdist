// version.cpp - FastDist version API implementation
#include <fastdist/version.h>

extern "C" int fd_version_major(void) {
    return FASTDIST_VERSION_MAJOR;
}

extern "C" int fd_version_minor(void) {
    return FASTDIST_VERSION_MINOR;
}

extern "C" int fd_version_patch(void) {
    return FASTDIST_VERSION_PATCH;
}
