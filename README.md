# Finetuning LLM Models

## **1. Model Quantization**

### Definition : Conversion from higher memory format to a lower memory format

Ex : ->In Neural networks which focus on weights are in matrix.Every value is stored in the memory in the form of 32 bits(FP(full precision/single precision) 32). So,we can convert 32bit to int 8 and then download or use the model(Inferencing becomes easy) so that it can be used in mobile phones,edge devices,smart watches etc.Since we are converting there is an loss of informatn which means loss of accuracy. We can overcome it..

* ### Full Precision/Half Precision

   ->DATA => Weights and Parameters
  
   ->FP32 bit (full precision) => FP16 bit(half precision)
  
* ### Calibration(squeezing) - Model Quantization

   How to perform Quantization :
  
   -> Symmetric Quantization       ->Asymmetric Quantization

     #### Symmetric Quantization :

      ->Batch Normalization(applied during forward or backward propogation so that all our weights are zero centred) is a technique of symmetric quantization

      -> * Symmetric Unsignedint8 Quantisation : Lets say we have a floating point number b/w  [0.0,......,1000.0]->numbers stored in form of 32 bits,we need to convert it to unint8(2^8 (0-255)).In Single precision floating point 32(ex:7.32) one bit is used for sign/unsign values(1),next 8 bits are stored for exponent(7),remaining 23 bits will be basically stored for mantissa(.32). If in half precision fp 16 one bit is used for sign/unsign values,next 5 bits are stored for exponent,remaining 10 bits will be basically stored for mantissa.

           Min max scalar : 0.0->0, 1000->255

           Scale = (xmax-xmin)/(qmax-qmin)
  
                 = 1000-0/255-0 = 3.92->scale factor

           round(250/3.92)=round(63.77)=64 (since symm distribution here zero point=0)

     #### Asymmetric Unsignedint8 Quantisation :
  
       ->[-20.0,....,1000.0] =>[0,...,255]

       ->Scale = (1000+20)/255 = 4.0->scale factor

       -> round(-20/4)=-5.0 as we need from [0,...255] we do -5+5(zero point)=0
        
  
* ### Modes of Quantization
   -> Post Training Quantization
   -> Quantization Aware Training

    #### Post Training Quantization(PTQ) :

       -> We already have pre trained model we apply calibration,we take weights data and convert into a quantized model then we can use this model for any use cases
 
    #### Quantization Aware Training(QAT) :

       -> If we perform caliibration and prepare a quantized model there is loss of data which leads to decrease in accuracy for any use cases.

       ->But in QAT we will be taking out trained model and we perform quantization,we will go ahead a perform fine tuning(we will take new training data,we will be fine tuning this model and create a quantized model)


## **2.FINE TUNING LLM-LoRA,QLoRA**

Whenever we have pretrained llm model(gpt4 turbo,gpt3.5),we save it as a basic model and this model is trained with huge amount of data.Further we take this model and do some amount of fine tuning on all weights of this specific model(full parameter fine tuning) and further if needed can perform domain specific fine tuning or specific task fine tuning

* ### Full parameter fine tuning 
  
   #### The major challenges we face are :

     -> Update all model weights
  
     -> Hardware resource constraints
  
         => for downstream task(ex : model monitoring, model influencing, GPU and RAM constraints) it is really difficult
  
   In order to overcome this task we use LoRA (LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS) and QLoRA (LoRA 2.0)

  * ### WHAT DOES LoRA DO?
 
    ->Instead of updating weights,it track changes(it will track changes of new weights based on fine tuning)

          Pretrained weights(n*n) + LoRA Tracked weights(n*n) = fine tuned weights(n*n)

                  W0              +        change in W        =     W0 + B*A             where B & A are decomposed matrices

          LoRA Tracked weights => matrix decomposition happens(same n*n matrix is saved into 2 smaller matrices) based on a parameter called rank (There can be a loss of precision)

    ->This will decrease number of trainable parameters

                    model parameters
    
          RANK   7B     13B     70B     180B

          1     167K    228K   529K     849K

          2     334K    456K    1M       2M

          8      1M      2M     4M       7M

          16     3M      4M     8M       14M

          512    86M    117M   270M     434M

    -> WHEN TO USE HIGH RANK : If the model wants to learn complex things then we can use high ranks


  * ### QLoRA : Quantized LoRA

    ->In case of LoRA if all the parameters are stored in 16bit we will try to convert this into 4bit so that we can reduce precison so that we don't require much more memory.

       
