/**
 * NNStreamer Android API
 * Copyright (C) 2019 Samsung Electronics Co., Ltd.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Library General Public License for more details.
 */

/**
 * @file	nnstreamer-native-singleshot.c
 * @date	10 July 2019
 * @brief	Native code for NNStreamer API
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include "nnstreamer-native.h"

/**
 * @brief Native method for single-shot API.
 */
jlong
Java_org_nnsuite_nnstreamer_SingleShot_nativeOpen (JNIEnv * env, jobject thiz,
    jstring model, jobject in, jobject out)
{
  pipeline_info_s *pipe_info = NULL;
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  const char *model_info = (*env)->GetStringUTFChars (env, model, NULL);

  single = NULL;
  in_info = out_info = NULL;

  if (in) {
    if (ml_tensors_info_create (&in_info) != ML_ERROR_NONE) {
      nns_loge ("Failed to create input tensors info.");
      goto done;
    }

    if (!nns_parse_tensors_info (pipe_info, env, in, in_info)) {
      nns_loge ("Failed to parse input tensor.");
      goto done;
    }
  }

  if (out) {
    if (ml_tensors_info_create (&out_info) != ML_ERROR_NONE) {
      nns_loge ("Failed to create output tensors info.");
      goto done;
    }

    if (!nns_parse_tensors_info (pipe_info, env, out, out_info)) {
      nns_loge ("Failed to parse output tensor.");
      goto done;
    }
  }

  /* supposed tensorflow-lite only for android */
  if (ml_single_open (&single, model_info, in_info, out_info,
          ML_NNFW_TYPE_ANY, ML_NNFW_HW_AUTO) != ML_ERROR_NONE) {
    nns_loge ("Failed to create the pipeline.");
    goto done;
  }

  pipe_info = nns_construct_pipe_info (env, thiz, single, NNS_PIPE_TYPE_SINGLE);

done:
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);

  (*env)->ReleaseStringUTFChars (env, model, model_info);
  return CAST_TO_LONG (pipe_info);
}

/**
 * @brief Native method for single-shot API.
 */
void
Java_org_nnsuite_nnstreamer_SingleShot_nativeClose (JNIEnv * env, jobject thiz,
    jlong handle)
{
  pipeline_info_s *pipe_info;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s*);

  nns_destroy_pipe_info (pipe_info, env);
}

/**
 * @brief Native method for single-shot API.
 */
jobject
Java_org_nnsuite_nnstreamer_SingleShot_nativeInvoke (JNIEnv * env,
    jobject thiz, jlong handle, jobject obj_data, jobject obj_info)
{
  pipeline_info_s *pipe_info;
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_data_s *in_data, *out_data;
  int status;
  jobject result = NULL;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s*);
  single = pipe_info->pipeline_handle;
  in_info = out_info = NULL;
  in_data = out_data = NULL;

  if ((in_data = g_new0 (ml_tensors_data_s, 1)) == NULL) {
    nns_loge ("Failed to allocate memory for input data.");
    goto done;
  }

  if (!nns_parse_tensors_data (pipe_info, env, obj_data, in_data)) {
    nns_loge ("Failed to parse input data.");
    goto done;
  }

  if (obj_info) {
    if (ml_tensors_info_create (&in_info) != ML_ERROR_NONE) {
      nns_loge ("Failed to create input tensors info.");
      goto done;
    }

    if (!nns_parse_tensors_info (pipe_info, env, obj_info, in_info)) {
      nns_loge ("Failed to parse input tensors info.");
      goto done;
    }

    status = ml_single_invoke_dynamic (single, in_data, in_info,
        (ml_tensors_data_h *) &out_data, &out_info);
  } else {
    status = ml_single_invoke (single, in_data, (ml_tensors_data_h *) &out_data);
  }

  if (status != ML_ERROR_NONE) {
    nns_loge ("Failed to get the result from pipeline.");
    goto done;
  }

  if (!nns_convert_tensors_data (pipe_info, env, out_data, &result)) {
    nns_loge ("Failed to convert the result to data.");
    result = NULL;
  }

done:
  ml_tensors_data_destroy ((ml_tensors_data_h) in_data);
  ml_tensors_data_destroy ((ml_tensors_data_h) out_data);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
  return result;
}

/**
 * @brief Native method for single-shot API.
 */
jobject
Java_org_nnsuite_nnstreamer_SingleShot_nativeGetInputInfo (JNIEnv * env,
    jobject thiz, jlong handle)
{
  pipeline_info_s *pipe_info;
  ml_single_h single;
  ml_tensors_info_h info;
  jobject result = NULL;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s*);
  single = pipe_info->pipeline_handle;

  if (ml_single_get_input_info (single, &info) != ML_ERROR_NONE) {
    nns_loge ("Failed to get input info.");
    goto done;
  }

  if (!nns_convert_tensors_info (pipe_info, env, info, &result)) {
    nns_loge ("Failed to convert input info.");
    result = NULL;
  }

done:
  ml_tensors_info_destroy (info);
  return result;
}

/**
 * @brief Native method for single-shot API.
 */
jobject
Java_org_nnsuite_nnstreamer_SingleShot_nativeGetOutputInfo (JNIEnv * env,
    jobject thiz, jlong handle)
{
  pipeline_info_s *pipe_info;
  ml_single_h single;
  ml_tensors_info_h info;
  jobject result = NULL;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s*);
  single = pipe_info->pipeline_handle;

  if (ml_single_get_output_info (single, &info) != ML_ERROR_NONE) {
    nns_loge ("Failed to get output info.");
    goto done;
  }

  if (!nns_convert_tensors_info (pipe_info, env, info, &result)) {
    nns_loge ("Failed to convert output info.");
    result = NULL;
  }

done:
  ml_tensors_info_destroy (info);
  return result;
}

/**
 * @brief Native method for single-shot API.
 */
jboolean
Java_org_nnsuite_nnstreamer_SingleShot_nativeSetTimeout (JNIEnv * env,
    jobject thiz, jlong handle, jint timeout)
{
  pipeline_info_s *pipe_info;
  ml_single_h single;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s*);
  single = pipe_info->pipeline_handle;

  if (ml_single_set_timeout (single, (unsigned int) timeout) != ML_ERROR_NONE) {
    nns_loge ("Failed to set the timeout.");
    return JNI_FALSE;
  }

  nns_logi ("Successfully set the timeout, %d milliseconds.", timeout);
  return JNI_TRUE;
}

/**
 * @brief Native method for single-shot API.
 */
jboolean
Java_org_nnsuite_nnstreamer_SingleShot_nativeSetInputInfo (JNIEnv * env,
    jobject thiz, jlong handle, jobject in)
{
  pipeline_info_s *pipe_info;
  ml_single_h single;
  ml_tensors_info_h in_info = NULL;
  jboolean ret = JNI_FALSE;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s*);
  single = pipe_info->pipeline_handle;

  if (ml_tensors_info_create (&in_info) != ML_ERROR_NONE) {
    nns_loge ("Failed to create input info handle.");
    return JNI_FALSE;
  }

  if (!nns_parse_tensors_info (pipe_info, env, in, in_info)) {
    nns_loge ("Failed to parse input tensor.");
    goto done;
  }

  if (ml_single_set_input_info (single, in_info) != ML_ERROR_NONE) {
    nns_loge ("Failed to set input info.");
    goto done;
  }

  ret = JNI_TRUE;

done:
  ml_tensors_info_destroy (in_info);
  return ret;
}
