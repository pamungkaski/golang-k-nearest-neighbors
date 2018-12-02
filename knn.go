// Package knn holds all K-Nearest Neighbors implementation.
package knn

import (
	"math"
	"sort"
)

type Data struct {
	// Name holds the data name(in this data is a number)
	Name string
	// X1 - X2 are the feature of this data.
	X1   float64
	X2   float64
	X3   float64
	X4   float64
	X5   float64
	// Y is the inference of the feature.
	Y    string
}

// Inference is a struct that holds the Inference distance between two data.
type Inference struct {
	Name     string
	Distance float64
}

// NearestNeighbors holds th contract for Nearest Neighbors algorithm.
type NearestNeighbors interface {
	Exec(obj Data, train []Data) string
	CalculateDistance(a, b Data) float64
}

// KNN is the implementation of NearestNeighbors with K modification.
type KNN struct {
	K int
}

// NewNearestNeighbors return new NearestNeighbors instance.
func NewNearestNeighbors(K int) NearestNeighbors {
	return &KNN{
		K: K,
	}
}

// CalculateDistance is a function to calculate the euclidean distance between two data feature.
func (k *KNN) CalculateDistance(a, b Data) float64 {
	return math.Sqrt(math.Pow(a.X1-b.X1, 2) + math.Pow(a.X2-b.X2, 2) + math.Pow(a.X3-b.X3, 2) + math.Pow(a.X4-b.X4, 2) + math.Pow(a.X5-b.X5, 2))
}

// Exec is a function to determine the inference of target data based on a train data by getting K closest distance.
func (k *KNN) Exec(a Data, train []Data) string {
	var dists []Inference
	inf := map[string]int{
		"0": 0,
		"1": 0,
		"2": 0,
		"3": 0,
	}

	// Calculate all distance between object to train data.
	for _, b := range train {
		var dist Inference
		dist.Name = b.Y
		dist.Distance = k.CalculateDistance(a, b)
		dists = append(dists, dist)
	}

	sort.Slice(dists, func(x, y int) bool {
		if dists[x].Distance == dists[y].Distance {
			return dists[x].Name < dists[y].Name
		}
		return dists[x].Distance < dists[y].Distance
	})

	// Get K nearest data.
	for j := 0; j < k.K; j++ {
		if dists[j].Name == "0" {
			inf["0"]++
		} else if dists[j].Name == "1" {
			inf["1"]++
		} else if dists[j].Name == "2" {
			inf["2"]++
		} else if dists[j].Name == "3" {
			inf["3"]++
		}
	}

	max := inf["0"]
	key := "0"
	for index, e := range inf {
		if max < e {
			max = e
			key = index
		}
	}

	// Return the most inference showed up in the first K data.
	return key
}
