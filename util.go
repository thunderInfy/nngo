package nngo

import (
	"fmt"
	"math/rand"
)

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

func Product[T int | float32 | float64](arr []T) (ret T) {
	ret = 1
	for _, t := range arr {
		ret *= t
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

func Max[T int | float32 | float64](values ...T) (ret T) {
	if len(values) == 0 {
		Panic(fmt.Errorf("error at least one value should be passed to max"))
	}
	ret = values[0]
	for i := 1; i < len(values); i += 1 {
		if ret < values[i] {
			ret = values[i]
		}
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

func RandomFloat64(randSource *rand.Rand, low float64, high float64) float64 {
	return randSource.Float64()*(high-low) + low
}

func IsNonIncreasing[T float64 | int](arr []T) bool {
	if len(arr) == 0 {
		return false // one can argue for both true or false, but let's go with false
	}
	for i := 1; i < len(arr); i++ {
		if arr[i] > arr[i-1] {
			return false
		}
	}
	return true
}

func DotProduct(a, b []float64) (res float64) {
	for i := range a {
		res += a[i] * b[i]
	}
	return res
}
