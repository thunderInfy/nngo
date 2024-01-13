package nngo

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// implement f(x,y,z) = (x+y)*z
func TestBackProp(t *testing.T) {

	var a, b Node

	x := InputSymbol("x", &a)
	y := InputSymbol("y", &a)
	z := InputSymbol("z", &b)
	f := OutputSymbol("f", &b)
	a = AddNode("a", &b, &x, &y)
	b = MultiplyNode("b", &f, &a, &z)

	basicGraph := Graph{
		Inputs:        [](*Node){&x, &y, &z},
		Output:        &f,
		Intermediates: [](*Node){&a, &b},
	}
	err := basicGraph.Forward([]float64{-2, 5, -4})
	Panic(err)
	assert.True(t, x.Val == -2)
	assert.True(t, y.Val == 5)
	assert.True(t, z.Val == -4)
	assert.True(t, a.Val == 3)
	assert.True(t, b.Val == -12)

	basicGraph.Backprop(1)
	assert.True(t, x.Grad == -4)
	assert.True(t, y.Grad == -4)
	assert.True(t, z.Grad == 3)
	assert.True(t, a.Grad == -4)
	assert.True(t, b.Grad == 1)
}
