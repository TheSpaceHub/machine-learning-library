#include "ml.h"
#include <stdlib.h>
typedef std::vector<double> vd;

int main()
{
    //test example: train model to detect, given a, b, c, if a * b > c
    Model myModel = Model(0.001, "squared_error");
    myModel.addLayer(5, 3, "sigmoid");
    myModel.addLayer(1, "tanh");
    myModel.print();
    std::vector<vd> train_input(0), train_target(0);
    for(int i = 0; i < 500; i++)
    {
        vd current = {};
        for(int j = 0;j<3; j++)
        {
            current.push_back(((double)(rand() % 100)) / 100);
        }
        std::cout << std::endl;
        train_input.push_back(current);
        train_target.push_back({(current[0] * current[1] > current[2])? (double)1 : (double)-1});
    }
    myModel.train(train_input, train_target, 20000);

    int count = 0;
    int a = 20;
    for(int i = 0; i < a; i++)
    {
        for(int j = 0; j < a; j++)
        {
            for(int k = 0; k < a; k++)
            {
                bool ans = (myModel.predict({(double) i / a, (double) j / a, (double) k / a / a})[0] >= 0);
                //std::cout << "(" << i << ", " << j << ", " << k << "): " << myModel.predict({(double) i / 10, (double) j / 10, (double) k / 20})[0] << std::endl;
                if(ans == (i * j > k)) count++;
            }
        }
    }
    //shows the rate of correct predictions
    std::cout << "Percentage: " << (double) count / a/a/a*100;
}