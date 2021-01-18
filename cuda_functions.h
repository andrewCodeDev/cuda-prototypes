
// MATRIX FUNCTIONS

__global__ void binaryAdd(const float *m1, const float *m2, float *m3, const int X)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  m3[row * X + col] = m1[row * X + col] + m2[row * X + col];
}

__global__ void binarySub(const float *m1, const float *m2, float *m3, const int X)
{

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  m3[row * X + col] = m1[row * X + col] - m2[row * X + col];
}

__global__ void binaryMul(const float *m1, const float *m2, float *m3, const int X)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  m3[row * X + col] = m1[row * X + col] * m2[row * X + col];
}

__global__ void binaryDiv(const float *m1, const float *m2, float *m3, const int X)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  m3[row * X + col] = m1[row * X + col] / m2[row * X + col];
}

__global__ void wgtUpdate(float *m1, const float *m2, const float lrn_rate, const int X)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  m1[row * X + col] -= m2[row * X + col] * lrn_rate;
}

__global__ void dotProduct(const float *m1, const int m1X, const float *m2, const int m2X, float *m3)
{

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float acm = 0.0f;

  for (int i = 0; i < m1X; ++i)
    acm += m1[row * m1X + i] * m2[i * m2X + col];

  m3[row * m2X + col] = acm;
}

__global__ void transpose(const float *m1, float *m2, const int X, const int Y)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  m2[col * Y + row] = m1[row * X + col];
}

__global__ void incrGPU(float *m1) // test function ...not used
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  m1[x] += 1.0f;
}

// ACTIVATION FUNCTIONS

__global__ void sigmoid(const float *m1, float *m2, const int X)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  m2[row * X + col] = 1.0f / (1.0f + expf(-m1[row * X + col]));
}

__global__ void sigmoidDeriv(const float *m1, float *m2, const int X)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  m2[row * X + col] = m1[row * X + col] * (1.0f - m1[row * X + col]);
}

__global__ void softmaxCol(const float *m1, float *m2, const int X)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float acm = 0.0f;

  for (int i = 0; i < X; ++i)
    acm += expf(m1[row * X + i]);

  m2[row * X + col] = expf(m1[row * X + col]) / acm;
}

__global__ void softmaxRow(const float *m1, float *m2, const int X, const int Y)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float acm = 0.0f;

  for (int i = 0; i < X; ++i)
    acm += expf(m1[i * X + col]);

  m2[row * X + col] = expf(m1[row * X + col]) / acm;
}

__global__ void squareMask(const float *m1, float *m2, const int X)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  int state = (row * X + col <= row * X + row);
  m2[row * X + col] = m1[row * X + col] * state + 1e-8f * 1 - state;
}

__global__ void crossEntropy(const float *m1, const float *trg, float *m2, const int X)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  m2[row * X + col] = -(trg[row * X + col] * logf(m1[row * X + col]) + (1.0f - trg[row * X + col]) * logf(1.0 - m1[row * X + col]));
}

__global__ void crossDeriv(const float *m1, const float *trg, float *m2, const int X)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  m2[row * X + col] = m1[row * X + col] - trg[row * X + col];
}

__global__ void softmaxDeriv(const float *m1, float *m2, const int X)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  m2[row * X + col] = m1[row * X + col] * (1.0f - m1[row * X + col]);
}

__global__ void relu(const float *m1, float *m2, const float relu_rate, const int X)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (m1[row * X + col] <= 0.0f)
    m2[row * X + col] = m1[row * X + col] * relu_rate;
  else
    m2[row * X + col] = m1[row * X + col];
}

__global__ void relu_deriv(const float *m1, float *m2, const float relu_rate, const int X)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (m1[row * X + col] <= 0.0f)
    m2[row * X + col] = relu_rate;
  else
    m2[row * X + col] = 1.0f;
}

// COMBINED ACTIVATION

__global__ void createATN(const float *m1, const int m1X, const float *m2, const int m2X,
                          float *m3, float *m4)
{

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float acm = 0.0f;

  for (int i = 0; i < m1X; ++i)
    acm += m1[row * m1X + i] * m2[i * m2X + col];

  m3[row * m2X + col] = acm;

  __syncthreads();

  acm = 0.0f;

  for (int i = 0; i < m2X; ++i)
    acm += expf(m3[i * m2X + col]);

  m4[row * m2X + col] = expf(m3[row * m2X + col]) / acm;
}

__global__ void createQKV(const float *INP, const int W_X, const int INP_X,
                          const float *WQ, float *AQ, const float *WK, float *AK, const float *WV, float *AV)
{

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float acmQ = 0.0f;
  float acmK = 0.0f;
  float acmV = 0.0f;

  for (int i = 0; i < W_X; ++i)
  {
    acmQ += WQ[row * W_X + i] * INP[i * INP_X + col];
    acmK += WK[row * W_X + i] * INP[i * INP_X + col];
    acmV += WV[row * W_X + i] * INP[i * INP_X + col];
  }

  AQ[row * INP_X + col] = acmQ;
  AK[row * INP_X + col] = acmK;
  AV[row * INP_X + col] = acmV;
}

__global__ void hiddenLayer(const float *m1, const int m1X, const float *m2, const int m2X,
                            float *m3, float *act_m, float *deriv_m, const float relu_rate)
{

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float acm = 0.0f;

  for (int i = 0; i < m1X; ++i)
    acm += m1[row * m1X + i] * m2[i * m2X + col];

  m3[row * m2X + col] = acm;

  if (m3[row * m2X + col] <= 0.0f)
  {
    act_m[row * m2X + col] = m3[row * m2X + col] * relu_rate;
    deriv_m[row * m2X + col] = relu_rate;
  }
  else
  {
    act_m[row * m2X + col] = m3[row * m2X + col];
    deriv_m[row * m2X + col] = 1.0f;
  }
}

__global__ void outputLayer(const float *m1, const int m1X, const float *m2, const int m2X,
                            float *m3, float *act_m, float *trg, float *error, float *deriv_m)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float acm = 0.0f;

  for (int i = 0; i < m1X; ++i)
    acm += m1[row * m1X + i] * m2[i * m2X + col];

  m3[row * m2X + col] = acm;

  //sigmoid
  acm = 1.0f / (1.0f + expf(-m3[row * m2X + col]));
  act_m[row * m2X + col] = acm;

  //cross entropy
  error[row * m2X + col] = -(trg[row * m2X + col] * logf(acm) + (1.0f - trg[row * m2X + col]) * logf(1.0 - acm));

  //entropy derivative
  deriv_m[row * m2X + col] = acm - trg[row * m2X + col];
}

__global__ void updateW2(const float *m1, const int m1X, const float *m2, const int m2X,
                         float *mW2, const float lrn_rate, float *m3)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  //store copy matrix for further backprop
  m3[row * m2X + col] = mW2[row * m2X + col];

  float acm = 0.0f;

  for (int i = 0; i < m1X; ++i)
    acm += m1[row * m1X + i] * m2[i * m2X + col];

  __syncthreads();
  mW2[row * m2X + col] -= acm * lrn_rate;
}

__global__ void updateGrad(float *m1, const int m1X, const float *m2, const int m2X,
                           float *m3, float *delta)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float acm = 0.0f;

  for (int i = 0; i < m1X; ++i)
    acm += m1[row * m1X + i] * m2[i * m2X + col];

  __syncthreads();
  m3[row * m2X + col] = acm * delta[row * m2X + col];
}

__global__ void updateW1(const float *m1, const int m1X, const float *m2, const int m2X,
                         float *mW1, const float lrn_rate, float *m3)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  m3[row * m2X + col] = mW1[row * m2X + col];

  float acm = 0.0f;

  for (int i = 0; i < m1X; ++i)
    acm += m1[row * m1X + i] * m2[i * m2X + col];

  __syncthreads();
  mW1[row * m2X + col] -= acm * lrn_rate;
}

__global__ void updateAWGTS(
    const float *dAQ, const float *dAK, const float *dAV,
    float *AQ, float *AK, float *AV, const float *INPT,
    const int m1X, const int m2X, const float lrn_rate)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float acmQ = 0.0f;
  float acmK = 0.0f;
  float acmV = 0.0f;

  for (int i = 0; i < m1X; ++i)
  {
    acmQ += dAQ[row * m1X + i] * INPT[i * m2X + col];
    acmK += dAK[row * m1X + i] * INPT[i * m2X + col];
    acmV += dAV[row * m1X + i] * INPT[i * m2X + col];
  }

  AQ[row * m2X + col] -= acmQ * lrn_rate;
  AK[row * m2X + col] -= acmK * lrn_rate;
  AV[row * m2X + col] -= acmV * lrn_rate;
}

__global__ void gradientATN(
    const float *m1, const float *m2, float *dAQ,
    const float *m3, const float *m4, float *dAK,
    const float *m5, const float *m6, float *dAV,
    const int m1X, const int m2X)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float acmQ = 0.0f;
  float acmK = 0.0f;
  float acmV = 0.0f;

  for (int i = 0; i < m1X; ++i)
  {
    acmQ += m1[row * m1X + i] * m2[i * m2X + col];
    acmK += m3[row * m1X + i] * m4[i * m2X + col];
    acmV += m5[row * m1X + i] * m6[i * m2X + col];
  }

  dAQ[row * m2X + col] = acmQ;
  dAK[row * m2X + col] = acmK;
  dAV[row * m2X + col] = acmV;
}

__global__ void softmaxGradient(
    const float *m1, const float *m2, const float *X2, float *m3,
    const int m1X, const int m2X)
{

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float acm = 0.0f;

  for (int i = 0; i < m1X; ++i)
    acm += m1[row * m1X + i] * m2[i * m2X + col];

  m3[row * m2X + col] = X2[row * m2X + col] * (1.0f - X2[row * m2X + col]) * acm;
}

//__global__ void softmaxDeriv(const float *m1, float *m2, const int X)
//{
//	int col = blockIdx.x * blockDim.x + threadIdx.x;
//	int row = blockIdx.y * blockDim.y + threadIdx.y;
//
//	m2[row * X + col] = m1[row * X + col] * (1.0f - m1[row * X + col]);
//}
//
//
//__global__ void binaryMul(const float *m1, const float *m2, float *m3, const int X)
//{
//	int col = blockIdx.x * blockDim.x + threadIdx.x;
//	int row = blockIdx.y * blockDim.y + threadIdx.y;
//
//	m3[row * X + col] = m1[row * X + col] * m2[row * X + col];
//}

// DELETE ME
//__global__ void testUpdateAWGTS
//(
//const float *dWQ, const float *dWK, const float *dWV,
// float *AQ, float *AK, float *AV, const int m2X, const float lrn_rate
//)
//{
//  int col = blockIdx.x * blockDim.x + threadIdx.x;
//  int row = blockIdx.y * blockDim.y + threadIdx.y;
//
// 	AQ[row * m2X + col] -= lrn_rate * dWQ[row * m2X + col];
// 	AK[row * m2X + col] -= lrn_rate * dWK[row * m2X + col];
// 	AV[row * m2X + col] -= lrn_rate * dWV[row * m2X + col];
//}

// CPU FUNCTIONS

template <typename T>
void print_matrix(T &vec, const int X, const int Y)
{
  std::cout << '\n';
  for (int i = 0; i < Y; ++i)
  {
    for (int j = 0; j < X; ++j)
      std::cout << vec[i * X + j] << ' ';
    std::cout << '\n';
  }
}
