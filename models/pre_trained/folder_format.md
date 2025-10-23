model_description.md : description of the model used, if possible source(s) for the architecture

Then however many training run files you need with the following format :
- training parameters (easy to just copy MATLAB trainingOptions arguments)
- Results (Validation fscore, Validation loss, time, iterations, as well as any extra interesting data (e.g. : was there any overfitting, interesting curve shape))