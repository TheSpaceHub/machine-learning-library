#include <iostream>
#include <vector>
typedef std::vector<double> vd;

class Node
{
private:
    int input_size;
public:
    vd weights;
    void test();
    Node(int i_size);
    double eval(vd input);
};

class Layer
{
private:
    int nodes_size;
    int input_size;
    std::string activation_function;
    const std::vector<std::string> allowed_activation_functions = {"l", "relu", "sigmoid", "lrelu", "exp", "tanh"};
public:
    vd func_vec(vd v);
    double func_val(double x);
    std::vector<Node*> nodes;
    Layer(int n_size, int i_size, std::string a_function);
    vd eval(vd input);
    vd feval(vd input);
    int get_nodes_size();
    int get_input_size();
    Node* get_node(int i);
    double derivative(double x);
};

class Model
{
private:
    double learning_rate;
    int layers_size;
    std::string loss_function;
    std::vector<Layer*> layers;
    vd eval(vd input);
public:
    Model(double l_rate, std::string l_function);
    void addLayer(int n_size, int i_size, std::string a_function = "");
    void addLayer(int n_size, std::string a_function = "");
    void print();
    double train_single(vd input, vd output);
    void train(std::vector<vd> inputs, std::vector<vd> outputs, int epochs = 5);
    double eval_loss(vd prediction, vd output);
    vd predict(vd input);
    double loss_derivative(vd prediction, vd target, int index);
};
