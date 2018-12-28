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

#include "lstm.h"
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(LSTM)

LSTM::LSTM()
{
    one_blob_only = false;
    support_inplace = false;
}

int LSTM::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    weight_data_size = pd.get(1, 0);

    return 0;
}

int LSTM::load_model(const ModelBin& mb)
{
    // The reason using "4" is because there are 4 attributes(elements) 
    // in a LSTM cell, and in ncnn parameters serialization method,  
    // the parameters corresponding with these elements were saved 
    // in continous way.
    // here is the W data saving form:
    //                                               |--------|--------|--------|--------|
    // h = size = weight_data_size / num_output / 4  |        |        |        |        |
    //                                               |--------|--------|--------|--------|
    //                                                gate 1 w gate 2 w gate 3 w gate 4 w
    int size = weight_data_size / num_output / 4;

    // WARNING:
    // the data in bin were saved continuously, so the order of calling .load() 
    // method should must be confirmed repeatedly which has to be consistance 
    // with the order of the LSTM cell elements saving in *.param file.
    
    /* original code:--------------------------------------
    // raw weight data
    // "hc" means hidden cell, 
    // these weights corresponding with the last time stamp's activation values, 
    // a_(t-1).
    weight_hc_data = mb.load(size, num_output * 4, 0);
    if (weight_hc_data.empty())
        return -100;
    // "xc" means X cell, 
    // these weights corresponding with the last layer's input values, 
    // x_(t).
    weight_xc_data = mb.load(size, num_output * 4, 0);
    if (weight_xc_data.empty())
        return -100;
    // "bc" means bias cell.
    bias_c_data = mb.load(4, num_output, 0);
    if (bias_c_data.empty())
        return -100;
    */ //-----------------------------------------------------
    
    // Suiting for caffe model:
    // "xc" means X cell, 
    // these weights corresponding with the last layer's input values, 
    // x_(t).
    weight_xc_data = mb.load(size, num_output * 4, 0);
    if (weight_xc_data.empty())
        return -100;
    // "bc" means bias cell.
    bias_c_data = mb.load(4, num_output, 0);
    if (bias_c_data.empty())
        return -100;
    // raw weight data
    // "hc" means hidden cell, 
    // these weights corresponding with the last time stamp's activation values, 
    // a_(t-1).
    // NOTICE:
    // Changed the dim of "weight_hc_data".
    weight_hc_data = mb.load(num_output, num_output * 4, 0);
    if (weight_hc_data.empty())
        return -100;

    return 0;
}

int LSTM::forward(const std::vector<Mat>& bottom_blobs, 
                  std::vector<Mat>& top_blobs, 
                  const Option& opt) const
{
    // elemsize represents sizeof(data type).
    const Mat& input_blob = bottom_blobs[0];
    size_t elemsize = input_blob.elemsize;

    // T, 0 or 1 each
    const Mat& cont_blob = bottom_blobs[1];

    // RNN time point number.
    int T = input_blob.h;
    // Width.
    int size = input_blob.w;

    // NOTICE:
    // "4u" is the default elementsize, represents 4 bytes.
    
    // initial hidden state
    // The hidden means the output from this layer in the last time point.
    Mat hidden(num_output, 4u, opt.workspace_allocator);
    if (hidden.empty())
        return -100;
    hidden.fill(0.f);

    // internal cell state, this layer's output at current time point.
    Mat cell(num_output, 4u, opt.workspace_allocator);
    if (cell.empty())
        return -100;
    
    // 4 * num_output, "4u" is size of data type, "4" for there are 4 
    // elements in one LSTM cell, corresponding with 4 gates.
    Mat gates(4, num_output, 4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    Mat& top_blob = top_blobs[0];
    top_blob.create(num_output, T, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // unroll
    // "T" represents the time points in RNN.
    for (int t=0; t<T; t++) {
        // clip hidden by continuation indicator
        // h_cont_{t-1} = cont_t * h_{t-1}
        // h_cont_{t-1} = h_{t-1} if cont_t == 1
        //                0       otherwise
        // calculate hidden
        // gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c
        
        // "cout" represents "cell output", "h_cout" represents "hidden cout".
        const int cont = ((const int*)cont_blob)[t];
        const float* x = input_blob.row(t);
        
        // matrix multiplication.
        // "q" represents the w(width) of each "weight tensor".
        for (int q = 0; q < num_output; q++) {
            /* original codes:
            float h_cont = cont ? hidden[q] : 0.f;
            */

            const float* bias_c_data_ptr = (const float*)bias_c_data + 4 * q;
            float* gates_data = (float*)gates + 4 * q;
            
            const float* I_bias_c_data_ptr = (const float*)bias_c_data;
            const float* F_bias_c_data_ptr = (const float*)bias_c_data + num_output;
            const float* O_bias_c_data_ptr = (const float*)bias_c_data + 2 * num_output;
            const float* G_bias_c_data_ptr = (const float*)bias_c_data + 3 * num_output;

            // gate I F O G：
            // I: ?, F: forgetten gate, O: output gate, G: ?.
            // Now handeling these 4 gate for X and a_(t-1) seperately.
            // Using "const float* weight_hc_data_I = (const float*)weight_hc_data + weight_hc_data.w * q;" 
            //
            // as an example, the handeling logic is:
            //
            // step 1, "(const float*)weight_hc_data":
            //     Hidden built an pointer to weight_hc_data and hidden convert the pointer type to 
            //     float.
            //
            // step 2, "+ weight_hc_data.w * q":
            //     Let pointer do some shifting. "*.w" is the column size for target 2d tensor, that is, 
            //     the width. "q" is increasing during iteration, the combination of "*.w" and "q" 
            //     means which row of the tensor we want the pointer point to.
            //     Note, the reason of using this wierd way to get certain row is, actually, the "ncnn::Mat" 
            //     saving  all weight values in a "1-dim array", for example, here is a "ncnn::Mat":
            //         |---|---|---|---|---|---|---|
            //     suppose this ncnn::Mat has: w==3, h==7, (c==1), which means this is an 7 * 3 array. 
            //     But in memory, this array is saving in the form of a 1-dim array with length 7*3. 
            //     So when we want get the value of ARRAY_NAME[4, 2], we should use:
            //         *(ARRAY_ADDRESS + 4 * width + 2
            //     In special case, if we use bits operation, we should:
            //         *(ARRAY_ADDRESS + 4 * width * sizeof(element) + 2 * sizeof(element)
            //
            // step 3, "+ size * 0":
            //     after "(const float*)weight_hc_data", the pointer we get is a point to the first of the
            //     mentioned "1-dim array", after step 2, we shift the pointer to the element which corresponding 
            //     with "certain row's" 1st element, but this "certain row" include the weight value for 4
            //     gates, so at step 3 we should shift the pointer to the 1st element of certain gate's 1st 
            //     element of this "certain row".
            //
            // NOTICE:
            // It seems there were some logical bug in original codes, now fix them.
            /* original codes:
            const float* weight_hc_data_I = (const float*)weight_hc_data + weight_hc_data.w * q + size * 0;
            const float* weight_xc_data_I = (const float*)weight_xc_data + weight_xc_data.w * q + size * 0;
            const float* weight_hc_data_F = (const float*)weight_hc_data + weight_hc_data.w * q + size * 1;
            const float* weight_xc_data_F = (const float*)weight_xc_data + weight_xc_data.w * q + size * 1;
            const float* weight_hc_data_O = (const float*)weight_hc_data + weight_hc_data.w * q + size * 2;
            const float* weight_xc_data_O = (const float*)weight_xc_data + weight_xc_data.w * q + size * 2;
            const float* weight_hc_data_G = (const float*)weight_hc_data + weight_hc_data.w * q + size * 3;
            const float* weight_xc_data_G = (const float*)weight_xc_data + weight_xc_data.w * q + size * 3;
            */
            const float* weight_hc_data_I = 
                (const float*)weight_hc_data + weight_hc_data.w * q + num_output * 0;
            const float* weight_xc_data_I = 
                (const float*)weight_xc_data + weight_xc_data.w * q + size * 0;
            
            const float* weight_hc_data_F = 
                (const float*)weight_hc_data + weight_hc_data.w * q + num_output * 1;
            const float* weight_xc_data_F = 
                (const float*)weight_xc_data + weight_xc_data.w * q + size * 1;
            
            const float* weight_hc_data_O = 
                (const float*)weight_hc_data + weight_hc_data.w * q + num_output * 2;
            const float* weight_xc_data_O = 
                (const float*)weight_xc_data + weight_xc_data.w * q + size * 2;
            
            const float* weight_hc_data_G = 
                (const float*)weight_hc_data + weight_hc_data.w * q + num_output * 3;
            const float* weight_xc_data_G = 
                (const float*)weight_xc_data + weight_xc_data.w * q + size * 3;

            /* original code:
            float I = bias_c_data_ptr[0];
            float F = bias_c_data_ptr[1];
            float O = bias_c_data_ptr[2];
            float G = bias_c_data_ptr[3];
            */
            float I = I_bias_c_data_ptr[q];
            float F = F_bias_c_data_ptr[q];
            float O = O_bias_c_data_ptr[q];
            float G = G_bias_c_data_ptr[q];
            
            // This is a "for loop" in "for loop", 
            // during the iteration of the outer for loop, the value of "h_cont" will be
            // dynamically changed, and in the inner for loop, it will dynamically indexing 
            // the value of "x", which is the current time point's input of current layer, 
            // and in this way, we can finish matrix multiplication.
            
            /* original codes:
            for (int i = 0; i < size; i++) {
                I += weight_hc_data_I[i] * h_cont + weight_xc_data_I[i] * x[i];
                F += weight_hc_data_F[i] * h_cont + weight_xc_data_F[i] * x[i];
                O += weight_hc_data_O[i] * h_cont + weight_xc_data_O[i] * x[i];
                G += weight_hc_data_G[i] * h_cont + weight_xc_data_G[i] * x[i];
            }
            */
            for (int i = 0; i < size; i++) {
                I += weight_xc_data_I[i] * x[i];
                F += weight_xc_data_F[i] * x[i];
                O += weight_xc_data_O[i] * x[i];
                G += weight_xc_data_G[i] * x[i];
            }
            for (int i = 0; i < num_output; ++i) {
                // h_cont: hidden cell out.
                // the reason is, the ncnn::Mat's data can be initialized as 0.
                float h_cont = cont == 0? 0: hidden[i];
                    
                I += weight_hc_data_I[i] * h_cont;
                F += weight_hc_data_F[i] * h_cont;
                O += weight_hc_data_O[i] * h_cont;
                G += weight_hc_data_G[i] * h_cont;
            }

            gates_data[0] = I;
            gates_data[1] = F;
            gates_data[2] = O;
            gates_data[3] = G;
        }

        // lstm unit
        // sigmoid(I)
        // sigmoid(F)
        // sigmoid(O)
        // tanh(G)
        // c_t := f_t .* c_{t-1} + i_t .* g_t
        // h_t := o_t .* tanh[c_t]
        float* output_data = top_blob.row(t);
        
        for (int q = 0; q < num_output; q++) {
            float* gates_data = (float*)gates + 4 * q;

            float I = gates_data[0];
            float F = gates_data[1];
            float O = gates_data[2];
            float G = gates_data[3];

            I = 1.f / (1.f + exp(-I));
            F = cont ? 1.f / (1.f + exp(-F)) : 0.f;
            O = 1.f / (1.f + exp(-O));
            G = tanh(G);

            float cell2 = F * cell[q] + I * G;
            float H = O * tanh(cell2);

            cell[q] = cell2;
            hidden[q] = H;
            output_data[q] = H;
        }

        // no cell output here
    }

    return 0;
}

} // namespace ncnn
