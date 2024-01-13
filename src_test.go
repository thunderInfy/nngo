package nngo

import (
	"fmt"
	"testing"
)

// implement f(x,y,z) = (x+y)*z
func TestBackProp(t *testing.T) {

	var x, y, z, a, b, f Node

	x = Node{
		Output: &a,
	}
	y = Node{
		Output: &a,
	}
	a = Node{
		Op:     Add,
		Inputs: [](*Node){&x, &y},
		Output: &b,
	}
	z = Node{
		Output: &b,
	}
	b = Node{
		Op:     Multiply,
		Inputs: [](*Node){&a, &z},
		Output: &f,
	}
	f = Node{
		Inputs: [](*Node){&b},
	}

	graph := Graph{
		Inputs: [](*Node){
			&x, &y, &z,
		},
		Outputs: [](*Node){
			&f,
		},
	}
	fmt.Println(graph)
}
