package nngo

import (
	"fmt"
	"testing"
)

// implement f(x,y,z) = (x+y)*z
func TestBackProp(t *testing.T) {

	var a, b Node

	x := InputSymbol(&a)
	y := InputSymbol(&a)
	z := InputSymbol(&b)
	f := OutputSymbol(&b)

	a = NewNode(Add, [](*Node){&x, &y}, &b)
	b = NewNode(Multiply, [](*Node){&a, &z}, &f)

	basicGraph := Graph{
		Inputs:        [](*Node){&x, &y, &z},
		Outputs:       [](*Node){&f},
		Intermediates: [](*Node){&a, &b},
	}
	fmt.Println(basicGraph)
}
