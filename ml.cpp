#include <iostream>
#include <cmath>
#include <stdlib.h>
#include "ml.h"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wsign-compare"
typedef std::vector<double> vd;

//constructors
Node::Node(int i_size)
{
    input_size = i_size;
    weights.resize(i_size + 1);
    for(int i = 0; i <= i_size; i++)
        weights[i] = (double)rand() / (double) RAND_MAX;
}

Layer::Layer(int n_size, int i_size, std::string a_function)
{
    nodes_size = n_size;
    input_size = i_size;
    nodes.resize(n_size);
    for(int i = 0; i < n_size; i++)
    {
        nodes[i] = new Node(i_size);
    }
    if (a_function == "") activation_function = "l";
    else
    {
        for(int i = 0; i < allowed_activation_functions.size(); i++)
        {
            if(a_function == allowed_activation_functions[i])
            {
                activation_function = a_function;
                return;
            }
        }
        std::cerr << "The following activation function does not exist: " + a_function << std::endl;
    }
}

Model::Model(double l_rate, std::string l_function)
{
    srand((unsigned)12345678);
    learning_rate = l_rate;
    layers.resize(0);
    layers_size = 0;
    loss_function = l_function;
}

//evaluating functions
double Node::eval(vd input)
{
    double evaluation = weights[0];
    if(input.size() != input_size)
        std::cerr << "Input size and input provided do not match." << std::endl;
    for(int i = 1; i <= input_size; i++)
        evaluation += input[i - 1] * weights[i];
    return evaluation;
}

vd Layer::eval(vd input)
{
    vd evaluated_nodes(nodes_size);
    for(int i = 0; i < nodes_size; i++)
        evaluated_nodes[i] = nodes[i]->eval(input);
    return evaluated_nodes;
}

vd Layer::feval(vd input)
{
    vd evaluated_nodes(nodes_size);
    for(int i = 0; i < nodes_size; i++)
        evaluated_nodes[i] = func_val(nodes[i]->eval(input));
    return evaluated_nodes;
}

vd Model::eval(vd input)
{
    vd current_vector = input;
    for(int i = 0; i < layers_size; i++)
        current_vector = layers[i]->feval(current_vector);
    return current_vector;
}


//NODE FUNCTIONS

//LAYER FUNCTIONS
//access private elements
int Layer::get_nodes_size()
{
    return nodes_size;
}
int Layer::get_input_size()
{
    return input_size;
}
Node* Layer::get_node(int i)
{
    return nodes[i];
}

//derivative for activation function
double Layer::derivative(double x)
{
    if(activation_function == "l") return 1;
    if(activation_function == "relu") return (x > 0)? 1 : 0;
    if(activation_function == "lrelu") return (x > 0)? 1 : 0.1;
    if(activation_function == "sigmoid") return std::exp(-x) / ( (1 + std::exp(-x)) * (1 + std::exp(-x)) );
    if(activation_function == "exp") return std::exp(x);
    if(activation_function == "tanh") return 1 - std::tanh(x) * std::tanh(x);
    return 0;
}

//pass through function
double Layer::func_val(double x)
{
    if(activation_function == "l") return x;
    if(activation_function == "relu") return (x > 0)? x : 0;
    if(activation_function == "lrelu") return (x > 0)? x : 0.1* x;
    if(activation_function == "sigmoid") return 1 / (1 + std::exp(-x));
    if(activation_function == "exp") return std::exp(x);
    if(activation_function == "tanh") return std::tanh(x);
    return 0;
}
//pass through function linear combination of nodes
vd Layer::func_vec(vd v)
{
    for(int i = 0; i < v.size(); i++)
        v[i] = func_val(v[i]);
    return v;
}

//MODEL FUNCTIONS
//view model
void Model::print()
{
    for(int i = 0; i < layers.size(); i++)
    {
        std::cout << "Layer " << i << ": " << layers[i]->get_nodes_size() << " nodes, input is " << layers[i]->get_input_size() << "\n";
        Layer current_layer = *(layers[i]);
        for(int j = 0; j < current_layer.get_nodes_size(); j++)
        {
            std::cout << "  Node " << j << " with weights ";
            for(int k = 0; k <= current_layer.get_input_size(); k++)
                std::cout << current_layer.get_node(j)->weights[k] << " ";
            std::cout << std::endl;
        }
    }
}

//add layer to model
void Model::addLayer(int n_size, int i_size, std::string a_function)
{
    layers.push_back(
        new Layer(n_size, i_size, a_function)
    );
    layers_size++;
}
void Model::addLayer(int n_size, std::string a_function)
{
    if(layers.size() == 0)
        std::cerr << "Input size not specified." << std::endl;
    int i_size = layers[layers.size() - 1]->get_nodes_size();
    layers.push_back(
        new Layer(n_size, i_size, a_function)
    );
    layers_size++;
}

//predict output
vd Model::predict(vd input)
{
    return eval(input);
}

//train model
void Model::train(std::vector<vd> inputs, std::vector<vd> outputs, int epochs)
{
    for(int i = 0; i < epochs; i++)
    {
        double av_loss = 0;
        for(int j = 0; j < inputs.size(); j++)
        {
            av_loss += train_single(inputs[j], outputs[j]) / (double) inputs.size();            
        }
        if((i + 1) % 5 == 0)
            std::cout << "Epoch " << i + 1 << " completed with average loss " << av_loss << "\n";
        //this->print();
    }
}

//train single example
double Model::train_single(vd input, vd target)
{
    //evaluate nodes here to keep track of each value before 
    std::vector<vd> n(layers_size + 1);
    n[0] = input;
    n[1] = layers[0]->eval(n[0]);
    for(int i = 2; i <= layers_size; i++)
        n[i] = layers[i-1]->eval(layers[i - 2]->func_vec(n[i - 1]));
    vd prediction = n[layers_size];
    double loss = eval_loss(prediction, target);

    //backpropagation

    //get deltas
    std::vector<vd> delta(layers_size, vd (0));
    for(int i = layers_size - 1; i >= 0; i--)
    {
        delta[i].resize(layers[i]->get_nodes_size());
        if(i == layers_size - 1)
        {
            for(int j = 0; j < delta[i].size(); j++)
            {
                delta[i][j] = loss_derivative(prediction, target, j) * layers[i]->derivative(n[i + 1][j]);
            }
        }
        else
        {
            for(int j = 0; j < delta[i].size(); j++)
            {
                double delta_sum = 0;
                for(int l = 0; l < layers[i + 1]->get_nodes_size(); l++)
                {
                    delta_sum += layers[i + 1]->nodes[l]->weights[j + 1] * delta[i + 1][l];
                }
                delta[i][j] = delta_sum * layers[i]->derivative(n[i + 1][j]);
            }
        }
    }

    //change weights
    //bias of input layer
    for(int l = 0; l < layers[0]->get_nodes_size(); l++)
    {
        layers[0]->nodes[l]->weights[0] -= learning_rate * delta[0][l];
    }

    //rest of input layer
    for(int j = 0; j < layers[0]->get_input_size(); j++)
    {
        for(int l = 0; l < layers[0]->get_nodes_size(); l++)
        {
            layers[0]->nodes[l]->weights[j + 1] -= learning_rate * delta[0][l] * n[0][j];
        }
    }

    //rest of layers
    for(int i = 1; i < layers_size; i++)
    {
        
        for(int l = 0; l < layers[i]->get_nodes_size(); l++)
        {
            //bias
            layers[i]->nodes[l]->weights[0] -= learning_rate * delta[i][l];
        
            //rest
            for(int j = 0; j < layers[i - 1]->get_nodes_size(); j++)
                layers[i]->nodes[l]->weights[j + 1] -= learning_rate * delta[i][l] * layers[i - 1]->func_val(n[i + 1][j]);

        }   
    }

    //reuse information for analytics
    return loss;

}

//get loss
double Model::eval_loss(vd prediction, vd output)
{
    if(prediction.size() != output.size()) std::cerr << "Prediction and output do not have same size\n";
    if(prediction.size() == 0) std::cout << "FUCKKKKKK\n\n\n\n";
    if(loss_function == "mean_squared_error")
    {
        double loss = 0;
        for(int i = 0; i < prediction.size(); i++)
            loss += (prediction[i] - output[i]) * (prediction[i] - output[i]);
        return loss / (double)prediction.size();
    }
    if(loss_function == "squared_error")
    {
        double loss = 0;
        for(int i = 0; i < prediction.size(); i++)
            loss += (prediction[i] - output[i]) * (prediction[i] - output[i]);
        return loss;
    }
    return 0;
}

//get loss derivative
double Model::loss_derivative(vd prediction, vd target, int index)
{
    if(loss_function == "mean_squared_error")
    {
        return 2 * (prediction[index] - target[index]) / (double) target.size();
    }
    if(loss_function == "squared_error")
    {
        return 2 * (prediction[index] - target[index]);
    }
    return 0;
}