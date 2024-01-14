package nngo

func Panic(err error) {
	if err == nil {
		return
	}
	panic(err.Error())
}

func Map[T, R any](arr []T, mapFunc func(t T) R) (ret []R) {
	ret = make([]R, len(arr))
	for i := range arr {
		ret[i] = mapFunc(arr[i])
	}
	return
}

func Sum[T int | float32 | float64](arr []T) (ret T) {
	ret = 0
	for _, t := range arr {
		ret += t
	}
	return
}

func ToPtrs[T any](arr []T) (ret [](*T)) {
	ret = make([](*T), len(arr))
	for i := range arr {
		ret[i] = &arr[i]
	}
	return
}
