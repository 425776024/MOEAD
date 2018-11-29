#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <fstream>
using namespace std;
int main()
{
    int m; // the number of objectives
    double stepsize;
    double H; // H = 1 / stepsize
    cout << "Please input the number of objectives (m): \n";
    cin >> m;
    cout << "Please input the stepsize (1/H): \n";
    cin >> stepsize;
    H = 1 / stepsize;
    cout << "H = " << H << endl;
    vector<int> sequence;
    for (unsigned int i = 0; i < H; i++) // the number of zero is (H)
    {
        sequence.push_back(0);
    }
    for (unsigned int i = 0; i < (m - 1); i++) // the number of 1 is (H + m - 1 - (m - 1))
    {
        sequence.push_back(1);
    }
     
    vector< vector<double> > ws;
    do 
    {
        int s = -1;
        vector<double> weight;
        for (unsigned int i = 0; i < (H + m - 1); i++)
        {
            if (sequence[i] == 1)
            {
                double w = i - s;
                w = (w - 1) / H;
                s = i;
                weight.push_back(w);
            }
        }
        double w = H + m - 1 - s;
        w = (w - 1) / H;
        weight.push_back(w);
        ws.push_back(weight);
    } while (next_permutation(sequence.begin(), sequence.end()));
    ofstream outfile("weight.txt");
    for (unsigned int i = 0; i < ws.size(); i++)
    {
        for (unsigned int j = 0; j < ws[i].size(); j++)
        {
            outfile << ws[i][j] << " ";
        }
        outfile << "\n";
    }
    return 0;
}
