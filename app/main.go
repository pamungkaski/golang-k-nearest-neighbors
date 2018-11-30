package main

import (
	"strconv"
	"os"
	"encoding/csv"
	"bufio"
	"github.com/pamungkaski/golang-k-nearest-neighbors"
	"io"
	"log"
	"fmt"
	"time"
	"math/rand"
)

func TakeRandom(data []knn.Data) ([]knn.Data, []knn.Data) {
	r := rand.New(rand.NewSource(time.Now().Unix()))
	val := make([]knn.Data, 200)
	perm := r.Perm(200)
	for i, randIndex := range perm {
		val[i] = data[randIndex]
	}
	for _, i := range perm {
		data[i] = data[len(data)-1] // Replace it with the last one.
		data = data[:len(data)-1]
	}
	return val, data
}

func StringtoData(line []string) (knn.Data) {
	var dt knn.Data
	var err error
	dt.Name = line[0]
	dt.X1, err = strconv.ParseFloat(line[1], 64)
	if err != nil {
		log.Fatal(err)
	}
	dt.X2, err = strconv.ParseFloat(line[2], 64)
	if err != nil {
		log.Fatal(err)
	}
	dt.X3, err = strconv.ParseFloat(line[3], 64)
	if err != nil {
		log.Fatal(err)
	}
	dt.X4, err = strconv.ParseFloat(line[4], 64)
	if err != nil {
		log.Fatal(err)
	}
	dt.X5, err = strconv.ParseFloat(line[5], 64)
	if err != nil {
		log.Fatal(err)
	}
	dt.Y = line[6]

	return dt
}

func main()  {
	///// DATA READING
	csvFile, _ := os.Open("DataTrain_Tugas3_AI.csv")
	reader := csv.NewReader(bufio.NewReader(csvFile))
	defer csvFile.Close()

	// K-Fold Cross Validation
	var data []knn.Data
	reader.Read()
	for  {
		line, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}

		dt := StringtoData(line)
		data = append(data, dt)
	}

	val, tren := TakeRandom(data)
	best := 1
	acc := 0.0

	for i := 1; i <= 200;  i++{
		kalg := knn.NewNearestNeighbors(i)
		right := 0

		for v:= 0; v < 200; v++ {
			a := val[v]
			kwey := kalg.Exec(a, tren)
			if kwey == a.Y {
				right++
			}
		}

		if float64(right)/200.0000 > acc {
			acc = float64(right)/200.0000
			best = i
		}
	}

	// Best K
	fmt.Println(best, acc)

	///// TRAIN DATA READING
	csvFile, _ = os.Open("DataTrain_Tugas3_AI.csv")
	reader = csv.NewReader(bufio.NewReader(csvFile))
	defer csvFile.Close()
	var train []knn.Data
	reader.Read()
	for  {
		line, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}

		dt := StringtoData(line)
		train = append(train, dt)
	}

	///// Test DATA READING
	csvFile, _ = os.Open("DataTest_Tugas3_AI.csv")
	reader = csv.NewReader(bufio.NewReader(csvFile))
	defer csvFile.Close()
	var test []knn.Data
	reader.Read()
	for  {
		line, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}

		dt := StringtoData(line)
		test = append(test, dt)
	}

	kalg := knn.NewNearestNeighbors(12)
	for index, a := range test {
		test[index].Y = kalg.Exec(a, train)
	}

	file, _ := os.Create("TebakanTugas3.csv")
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	///////// INSERTING DATA
	head := []string{
		"No",
		"X1",
		"X2",
		"X3",
		"X4",
		"X5",
		"Y",
	}
	if err := writer.Write(head); err != nil {
		log.Fatalln("error writing record to csv:", err)
	}

	for _, t := range test {
		csvData := []string{
			fmt.Sprintf("%s", t.Name),
			fmt.Sprintf("%.6f", t.X1),
			fmt.Sprintf("%.6f", t.X2),
			fmt.Sprintf("%.6f", t.X3),
			fmt.Sprintf("%.6f", t.X4),
			fmt.Sprintf("%.6f", t.X5),
			fmt.Sprintf("%s", t.Y),
		}
		if err := writer.Write(csvData); err != nil {
			log.Fatalln("error writing record to csv:", err)
		}
	}
}
