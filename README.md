# Finetuning LLM Models

## **1. Model Quantization**

### Definition : Conversion from higher memory format to a lower memory format

Ex : ->In Neural networks which focus on weights are in matrix.Every value is stored in the memory in the form of 32 bits(FP(full precision/single precision) 32). So,we can convert 32bit to int 8 and then download or use the model(Inferencing becomes easy) so that it can be used in mobile phones,edge devices,smart watches etc.Since we are converting there is an loss of informatn which means loss of accuracy. We can overcome it..

* ### Full Precision/Half Precision

   ->DATA => Weights and Parameters
  
   ->FP32 bit (full precision) => FP16 bit(half precision)
  
* ### Calibration - Model Quantization

   How to perform Quantization :
  
   -> Symmetric Quantization       ->Asymmetric Quantization

     #### Symmetric Quantization :

      ->Batch Normalization(applied during forward or backward propogation so that all our weights are zero centred) is a technique of symmetric quantization

      ->* Symmetric Unsignedint8 Quantisation : Lets say we have a floating point number b/w  [0.0,......,1000.0]->numbers stored in form of 32 bits,we need to convert it to unint8(2^8 (0-255)).In Single precision floating point 32(ex:7.32) one bit is used for sign/unsign values(1),next 8 bits are stored for exponent(7),remaining 23 bits will be basically stored for mantissa(.32). If in half precision fp 16 one bit is used for sign/unsign values,next 5 bits are stored for exponent,remaining 10 bits will be basically stored for mantissa(.32).

           Min max scalar : 0.0->0, 1000->255

           Scale = xmax-xmin/qmax-qmin
  
                 = 1000-0/255-0 = 3.92->scale factor

           round(250/3.92)=round(63.77)=64 (since symm distribution here zero point=0)

      ->* Asymmetric Unsignedint8 Quantisation : [-20.0,....,1000.0] =>[0,...,255]

          Scale = 1000+20/255 = 4.0->scale factor

           round(-20/4)=-5.0 as we need from [0,...255] we do -5+5(zero point)=0
        
  
* ### Modes of Quantization
   -> Post Training Quantization
   -> Quantization Aware Training
