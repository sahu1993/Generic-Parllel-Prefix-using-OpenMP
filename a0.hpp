/*  YOUR_FIRST_NAME - Shivam
 *  YOUR_LAST_NAME - Sahu
 *  YOUR_UBIT_NAME - ssahu3
 */

#include <omp.h>
#include <math.h>

#ifndef A0_HPP
#define A0_HPP

template <typename T, typename Op>
void omp_scan(int n, const T* in, T* out, Op op) {

  int chunk;
  int *l_Ele_Chunk;

  #pragma omp parallel shared(in,out,chunk)
  {
    const int threadID = omp_get_thread_num();
    const int totalThread = omp_get_num_threads();
    int sum = 0;
    #pragma omp single
    {
      chunk  = ceil(n/(float)totalThread);
      //std::cout << "Computation start with threads:" << totalThread << "\n";
      l_Ele_Chunk = new int[totalThread+1];
      l_Ele_Chunk[0] = 1;
    }

    #pragma omp for schedule(static,chunk)
    for(int i=0; i<n; i++) {
      if(i%chunk == 0){
        sum = in[i];
        out[i] = sum;
      }else{
        sum = op(sum,in[i]);
        out[i] = sum;
      }
    }

    l_Ele_Chunk[threadID+1] = sum;
    #pragma omp barrier

    int shift = l_Ele_Chunk[1];
    for(int i=2; i<(threadID+1); i++) {
      shift = op(shift,l_Ele_Chunk[i]);
    }

    #pragma omp for schedule(static,chunk)
    for(int i=0; i<n; i++) {
      if((chunk - i) > 0){
        //do nothing
      }else{
      out[i] = op(out[i],shift);
      }
    }
  }

} // omp_scan

#endif // A0_HPP
