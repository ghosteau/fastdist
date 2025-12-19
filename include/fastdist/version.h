// Version header for the FastDist library
#pragma once

// Version denoted as major.minor.patch
#define FASTDIST_VERSION_MAJOR 0
#define FASTDIST_VERSION_MINOR 1
#define FASTDIST_VERSION_PATCH 0

#ifdef __cplusplus
extern "C" {
#endif

// Returns the major, minor, and patch version numbers respectively
int fd_version_major(void);
int fd_version_minor(void);
int fd_version_patch(void);

#ifdef __cplusplus
}
#endif
