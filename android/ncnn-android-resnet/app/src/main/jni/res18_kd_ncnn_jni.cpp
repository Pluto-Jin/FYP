// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>
#include <math.h>

// ncnn
#include "net.h"
#include "benchmark.h"

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static ncnn::Net resnet;

static std::vector<std::string> split_string(const std::string& str, const std::string& delimiter)
{
    std::vector<std::string> strings;

    std::string::size_type pos = 0;
    std::string::size_type prev = 0;
    while ((pos = str.find(delimiter, prev)) != std::string::npos)
    {
        strings.push_back(str.substr(prev, pos - prev));
        prev = pos + 1;
    }

    // To get the last substring (or only, if delimiter is not found)
    strings.push_back(str.substr(prev));

    return strings;
}

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "JNI_OnLoad");

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "JNI_OnUnload");

    ncnn::destroy_gpu_instance();
}

// public native boolean Init(AssetManager mgr);
JNIEXPORT jboolean JNICALL Java_com_tencent_resnetncnn_ResNetNcnn_Init(JNIEnv* env, jobject thiz, jobject assetManager)
{
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;

    // use vulkan compute
    if (ncnn::get_gpu_count() != 0)
        opt.use_vulkan_compute = true;

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    resnet.opt = opt;

    // init param
    {
        int ret = resnet.load_param(mgr, "res18_kd.param");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "ResNetNcnn", "load_param_bin failed");
            return JNI_FALSE;
        }
    }

    // init bin
    {
        int ret = resnet.load_model(mgr, "res18_kd.bin");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "load_model failed");
            return JNI_FALSE;
        }
    }

    return JNI_TRUE;
}

// public native String Detect(Bitmap bitmap, boolean use_gpu);
JNIEXPORT jstring JNICALL Java_com_tencent_resnetncnn_ResNetNcnn_Detect(JNIEnv* env, jobject thiz, jobject bitmap, jboolean use_gpu)
{
    double abb = 1.0;
    __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "%.2fms   detect", abb);
    if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0)
    {
        return env->NewStringUTF("no vulkan capable gpu");
    }

    double start_time = ncnn::get_current_time();

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    int width = info.width;
    int height = info.height;
    __android_log_print(ANDROID_LOG_DEBUG, "width","%d",width);
    __android_log_print(ANDROID_LOG_DEBUG, "height","%d",height);
//    if (width != 227 || height != 227)
//        return NULL;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;

    // ncnn from bitmap
    ncnn::Mat in = ncnn::Mat::from_android_bitmap(env, bitmap, ncnn::Mat::PIXEL_BGR2RGB);

    // squeezenet
    std::vector<float> cls_scores;
    {
        const float mean_vals[3] = {255.0*0.485f, 255.0*0.456f, 255.0*0.406f};
        const float norm_vals[3] = {1.0/255.0/0.229f, 1.0/255.0/0.224f, 1.0/255.0/0.225f};
        in.substract_mean_normalize(mean_vals, norm_vals);
        ncnn::Extractor ex = resnet.create_extractor();

        ex.set_vulkan_compute(use_gpu);

        ex.input("input.1", in);

        ncnn::Mat out;
        ex.extract("191", out);

        cls_scores.resize(out.w);
        for (int j=0; j<out.w; j++)
        {
            cls_scores[j] = out[j];
        }
    }

    // return top class
    int top_class = 0;
    float max_score = 0.f;
    float exp_sum = 0.f;
    for (size_t i=0; i<cls_scores.size(); i++)
    {
        float s = cls_scores[i];
        exp_sum += exp(s);
//         __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "%d %f", i, s);
        if (s > max_score)
        {
            top_class = i;
            max_score = s;
        }
    }
    float prob = exp(max_score) / exp_sum;

    std::string class_name[10] = {"airplane", "automobile", \
        "bird", "cat", "deer", "dog", "frog", \
        "horse", "ship", "truck"};
    char tmp[32],tmp2[32];
    sprintf(tmp, "%.3f", max_score);
    sprintf(tmp2, "%.3f", prob);
    std::string result_str = " " + class_name[top_class] + " = " + tmp2 + " (logit=" + tmp +")";

    jstring result = env->NewStringUTF(result_str.c_str());

    double elasped = ncnn::get_current_time() - start_time;
    __android_log_print(ANDROID_LOG_DEBUG, "ResNetNcnn", "%.2fms   detect", elasped);

    return result;
}

}
