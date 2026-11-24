// version.cpp - probforge version API implementation
#include <probforge/version.h>

extern "C" int fd_version_major(void) { return PROBFORGE_VERSION_MAJOR; }

extern "C" int fd_version_minor(void) { return PROBFORGE_VERSION_MINOR; }

extern "C" int fd_version_patch(void) { return PROBFORGE_VERSION_PATCH; }
