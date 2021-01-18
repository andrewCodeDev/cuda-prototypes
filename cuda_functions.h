template <typename F, typename T>
std::vector<T> transform_vec(F f, const std::vector<T> &X)
{
	std::vector<T> Y;
	Y.reserve(X.size());
	std::transform(begin(X), end(X), back_inserter(Y), f);
	return Y;
}
