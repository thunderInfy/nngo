package nngo

func Panic(err error) {
	if err == nil {
		return
	}
	panic(err.Error())
}
