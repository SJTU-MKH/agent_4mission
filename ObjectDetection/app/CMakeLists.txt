# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

# Declares and names the project.

project("objectdetection")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

cmake_minimum_required(VERSION 3.4.1)
#set(OpenCV_DIR /home/khm/AndroidStudioProjects/test_cpp/open/cv341/native/jni)
#set(OpenCV_DIR src/main/cpp/openCV341/sdk/native/jni)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--exclude-libs,libippicv.a -Wl,--exclude-libs,libippiw.a")


#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY /home/khm/AndroidStudioProjects/cpp_cv/app/libs/x86)
#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/libs/x86)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/libs/x86)
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/opencv/sdk/native/jni)

#set(ANDROID_NDK_ABI_NAME armeabi-v7a)
#message(STATUS "路径:${CMAKE_CURRENT_SOURCE_DIR}/opencvnative3_4_2/jni}")
#find_library(jnigraphics-lib jnigraphics)
find_package(OpenCV REQUIRED)
find_library(jnigraphics-lib jnigraphics)

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    PKG_CONFIG_PATH: ${PKG_CONFIG_PATH}")
message(STATUS "    ANDROID_NDK_ABI_NAME: ${ANDROID_NDK_ABI_NAME}")

include_directories(
        include
        src/main/cpp
        src/main/cpp/include/opencv2
        src/main/cpp/qr
        src/main/cpp/qr/detector
        src/main/cpp/qr/scale
        src/main/cpp/qr/zxing
        src/main/cpp/qr/zxing/common
        src/main/cpp/qr/zxing/qrcode
        src/main/cpp/qr/zxing/common/binarizer
        src/main/cpp/qr/zxing/common/reedsolomon
        src/main/cpp/qr/zxing/qrcode/decoder
        src/main/cpp/qr/zxing/qrcode/detector
)

aux_source_directory(src/main/cpp/qr SOURCES0)
aux_source_directory(src/main/cpp/qr/detector SOURCES1)
aux_source_directory(src/main/cpp/qr/scale SOURCES2)
aux_source_directory(src/main/cpp/qr/zxing SOURCES3)
aux_source_directory(src/main/cpp/qr/zxing/common SOURCES4)
aux_source_directory(src/main/cpp/qr/zxing/qrcode SOURCES5)
aux_source_directory(src/main/cpp/qr/zxing/common/binarizer SOURCES6)
aux_source_directory(src/main/cpp/qr/zxing/common/reedsolomon SOURCES7)
aux_source_directory(src/main/cpp/qr/zxing/qrcode/decoder SOURCES8)
aux_source_directory(src/main/cpp/qr/zxing/qrcode/detector SOURCES9)
aux_source_directory(src/main/cpp SOURCES10)

list (APPEND SOURCES
        ${SOURCES0}
        ${SOURCES1}
        ${SOURCES2}
        ${SOURCES3}
        ${SOURCES4}
        ${SOURCES5}
        ${SOURCES6}
        ${SOURCES7}
        ${SOURCES8}
        ${SOURCES9}
        ${SOURCES10}
        )
add_library( # Sets the name of the library.
        objectdetection
        # Sets the library as a shared library.
        SHARED
        # Provides a relative path to your source file(s).
        ${SOURCES})

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib
        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        objectdetection
        # Links the target library to the log library
        # included in the NDK.
        jnigraphics
        ${log-lib}
        ${OpenCV_LIBS})

