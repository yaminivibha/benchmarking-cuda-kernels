#include <iostream>
#include <chrono>
#include <cstdlib> 

using namespace std;

int main(int argc, char* argv[]) {
  // correcting usage
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " K" << endl;
    return 1;
  }
  const int K = stoi(argv[1]);
  const int K_million = K * 1000000; 
  int* arr1 = new int[K_million];
  int* arr2 = new int[K_million];
  for (int i = 0; i < K_million; i++) {
    arr1[i] = (float)i;
    arr2[i] = (float)(K_million-i);   
  }
  // timing only the addition
  auto start_time = chrono::steady_clock::now();
  int* result = new int[K_million];
  for (int i = 0; i < K_million; i++) {
    result[i] = arr1[i] + arr2[i];
  }
  auto end_time = chrono::steady_clock::now();
  auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
  
  // printing result
  cout << "Runtime: " << duration << " ms" << endl;
  
  // Verify & report result
    for (i = 0; i < N; ++i) {
        float val = h_C[i];
        if (fabs(val - N) > 1e-5)
            break;
    }
    if (i == N)
        printf("PASSED\n");
    else
        printf("FAILED\n");

  // freeing memory
  free(arr1);
  free(arr2);
  free(result);
  return 0;
}