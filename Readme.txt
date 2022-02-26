This is the code we used to run our simulations and create our plots. 

Running the main file will run all the code used for the project. 

The main file accesses the experiments file where we declare the basic properties of our experiments. These include the data used, the number of runs and epochs, the parameter q for the moving average and the parameter s for the learning rate scheduler.

We then run the functions simulate runs that runs our simulations a fixed number of times for the two optimizers. We pass a function from the optimized_functions file where we specify the initial learning rate and the objective and the gradient.

Finally we use the functions in plots to create and save the evaluation of our simulations.