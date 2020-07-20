int a = 3;
int main(void) {
  // the result will be promoted as exit return
  int b = 2;
  return ((((a * b) + b) * (a + b)) + b);
}
