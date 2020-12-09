#include <bits/stdc++.h>

using namespace std;

// function to calculate dot product of two vectors
long double dot_product(vector<long double> A, vector<long double> B)
{
	if(A.size()!=B.size())
	{
		cout<<"Error"<<endl;
		return 0;
	}
	long double value = 0;
	for(int i=0; i<A.size(); i++)
	{
		value += A[i]*B[i];
	}
	return value;
}

int main()
{
	// taking inputs
	int V; // size of vocabulay
	int N; // size of hidden layer
	int iterations; // number of iterations
	int pairs; // number of word pairs
	long double learning_rate; // learning rate
	cin >> V >> N >> learning_rate >> iterations >> pairs;

	// word pairs to be trained
	vector<tuple<int, int, int> > word_pairs(pairs);

	for(int p=0; p<pairs; p++)
	{
		int pair_id, input_word, output_word;
		cin >> pair_id >> input_word >> output_word;

		word_pairs[p] = make_tuple(pair_id, input_word, output_word);
	}

	// input and output matrix
	vector<vector<long double> > W(V+1, vector<long double> (N+1,0.5));
	vector<vector<long double> > Wprime(N+1, vector<long double> (V+1,0.5));

	for(int i=0; i<=N; i++)
	{
		W[0][i] = 0;
		Wprime[i][0] = 0;
	}

	for(int i=0; i<=V; i++)
	{
		W[i][0] = 0;
		Wprime[0][i] = 0;
	}

	// training on all word pairs for given number of iterations
	for(int iter=0; iter<iterations; iter++)
	{
		for(int p=0; p<pairs; p++)
		{
			int pair_id = get<0>(word_pairs[p]);
			int input_word = get<1>(word_pairs[p]);
			int output_word = get<2>(word_pairs[p]);

			// one hot encoding of the input word
			vector<int> x(V+1,0);
			x[input_word] = 1;

			//forward propagation

			// hidden layer
			vector<long double> h(N+1);
			h = W[input_word];

			// output layer
			vector<long double> u(V+1);
			long double max_u = INT_MIN;
			for(int j=1; j<=V; j++)
			{
				vector<long double> vprime(N+1,0);
				for(int x=1; x<=N; x++)
					vprime[x] = Wprime[x][j];
				u[j] = dot_product(vprime,h);
				max_u = max(u[j],max_u);
			}

			// denominator for softmax
			long double sigma_u = 0;
			for(int j=1; j<=V; j++)
			{
				sigma_u += exp(u[j]-max_u);
			}

			// softmax output layer
			vector<long double> y(V+1);
			for(int j=1; j<=V; j++)
			{
				y[j] = exp(u[j]-max_u)/sigma_u;
			}

			//backward propagation
			vector<vector<long double> > WprimeOld = Wprime;

			// calculation error vector
			vector<long double> e(V+1,0);
			for(int j=1; j<=V; j++)
			{
				if(j==output_word)
					e[j] = y[j]-1;
				else
					e[j] = y[j];
			}

			// number of negative and non negative updates
			int negative_updates = 0;
			int non_negative_updates = 0;

			for(int i=1; i<=N; i++)
			{
				int j = output_word;
				long double update = learning_rate*e[j]*h[i];
				if(update > 0)
					negative_updates++;
				else
					non_negative_updates++;
				Wprime[i][j] = Wprime[i][j] - update;
			}

			// EH vector calculation
			vector<long double> EH(N+1,0);
			for(int i=1; i<=N; i++)
			{
				for(int j=1; j<=V; j++)
				{
					EH[i] += e[j]*WprimeOld[i][j];
				}
			}

			// updating the input matrix
			for(int i=1; i<=N; i++)
			{
				long double update = learning_rate*EH[i];
				if(update > 0)
					negative_updates++;
				else
					non_negative_updates++;
				W[input_word][i] = W[input_word][i] - update;
			}
			cout<<iter+1<<" "<<pair_id<<" "<<negative_updates<<" "<<non_negative_updates<<endl;
		}
	}

	// printing input and output matrix
	// for(int i=1; i<=V; i++)
	// {
	// 	for(int j=1; j<=N; j++)
	// 	{
	// 		cout<<W[i][j]<<" ";
	// 	}
	// 	cout<<endl;
	// }

	// for(int i=1; i<=N; i++)
	// {
	// 	for(int j=1; j<=V; j++)
	// 	{
	// 		cout<<Wprime[i][j]<<" ";
	// 	}
	// 	cout<<endl;
	// }

	return 0;
}