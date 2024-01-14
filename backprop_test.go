package nngo

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// f(x,y,z) = (x+y)*z
func TestBackProp1(t *testing.T) {

	var a, b Node

	x := InputSymbol("x", &a)
	y := InputSymbol("y", &a)
	z := InputSymbol("z", &b)
	f := OutputSymbol("f", &b)
	a = AddNode("a", &b, [](*Node){&x, &y})
	b = MultiplyNode("b", &f, &a, &z)

	graph, err := NewGraph([](*Node){&x, &y, &z}, &f, [](*Node){&a, &b})
	Panic(err)

	err = graph.Forward([]float64{-2, 5, -4})
	Panic(err)
	assert.True(t, x.Val == -2)
	assert.True(t, y.Val == 5)
	assert.True(t, z.Val == -4)
	assert.True(t, a.Val == 3)
	assert.True(t, b.Val == -12)

	graph.Backprop(1)
	assert.True(t, x.Grad == -4)
	assert.True(t, y.Grad == -4)
	assert.True(t, z.Grad == 3)
	assert.True(t, a.Grad == -4)
	assert.True(t, b.Grad == 1)
}

// f(a,b,c,x,y,z) = ax + by + cz = p + q + r
func TestBackProp2(t *testing.T) {

	var p, q, r, s Node

	a := InputSymbol("a", &p)
	x := InputSymbol("x", &p)
	p = MultiplyNode("p", &s, &a, &x)

	b := InputSymbol("b", &q)
	y := InputSymbol("y", &q)
	q = MultiplyNode("q", &s, &b, &y)

	c := InputSymbol("c", &r)
	z := InputSymbol("z", &r)
	r = MultiplyNode("q", &s, &c, &z)

	f := OutputSymbol("f", &s)

	s = AddNode("s", &f, [](*Node){&p, &q, &r})

	graph, err := NewGraph([](*Node){&a, &x, &b, &y, &c, &z}, &f, [](*Node){&p, &q, &r, &s})
	Panic(err)

	err = graph.Forward([]float64{-1, 4, 5, -6, 2, 7})
	Panic(err)
	assert.True(t, a.Val == -1)
	assert.True(t, x.Val == 4)
	assert.True(t, b.Val == 5)
	assert.True(t, y.Val == -6)
	assert.True(t, c.Val == 2)
	assert.True(t, z.Val == 7)
	assert.True(t, p.Val == -4)
	assert.True(t, q.Val == -30)
	assert.True(t, r.Val == 14)
	assert.True(t, s.Val == -20)

	graph.Backprop(1)
	assert.True(t, a.Grad == 4)
	assert.True(t, x.Grad == -1)
	assert.True(t, b.Grad == -6)
	assert.True(t, y.Grad == 5)
	assert.True(t, c.Grad == 7)
	assert.True(t, z.Grad == 2)
	assert.True(t, p.Grad == 1)
	assert.True(t, q.Grad == 1)
	assert.True(t, r.Grad == 1)
	assert.True(t, s.Grad == 1)
}
