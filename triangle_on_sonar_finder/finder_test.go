package triangle_on_sonar_finder

import (
	"image"
	"image/color"
	"image/draw"
	"os"
	"testing"

	"go.viam.com/test"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func openImage(fn string) (image.Image, error) {
	file, err := os.Open(fn)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	return img, err
}

func TestTriangleOnSonarFinder(t *testing.T) {
	scale := 0.5
	templates, err := loadTemplates(scale)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, len(templates), test.ShouldEqual, 13)

	for i, tmpl := range templates {
		t.Logf("Template %d: Original size: %v, Resized size: %dx%d",
			i, tmpl.originalSize, tmpl.kernelWidth, tmpl.kernelHeight)
		//if i == 4 { // save preprocessed template for debugging
		//	edgeImg := EdgeMatrixToGrayImage(tmpl.kernel)
		//	err = SaveImageAsPNG(edgeImg, "debug_template_edge.png")
		//	test.That(t, err, test.ShouldBeNil)
		//}
	}
	img, err := openImage("inputs/white_bg.png")
	test.That(t, err, test.ShouldBeNil)
	test.That(t, img.Bounds().Empty(), test.ShouldBeFalse)

	imgMatrix := ImageToMatrix(img, scale)

	//edgeImg2 := EdgeMatrixToGrayImage(imgMatrix)
	//err = SaveImageAsPNG(edgeImg2, "debug_edge_matrix.png") //save edge matrix for debugging
	//test.That(t, err, test.ShouldBeNil)

	detections := findTriangles(templates, imgMatrix, 2, 0.65, scale)

	//drawing detections on image
	rgbaImg := image.NewRGBA(img.Bounds())
	draw.Draw(rgbaImg, rgbaImg.Bounds(), img, image.Point{}, draw.Src)
	for i, det := range detections {
		t.Logf("Detection %d: Box=%v, Score=%f", i, det.BoundingBox(), det.Score())
		DrawBoundingBox(rgbaImg, *det.BoundingBox(), color.RGBA{255, 0, 0, 255}, 2, float32(det.Score())) // red, thickness 2
	}
	err = SaveImageAsPNG(rgbaImg, "detections/debug_detections_2.png") //save detections with bb for debugging
	test.That(t, err, test.ShouldBeNil)

	//verify number of detections
	test.That(t, len(detections), test.ShouldEqual, 3)
}

func BenchmarkTriangls(t *testing.B) {
	scale := 0.5
	templates, err := loadTemplates(scale)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, len(templates), test.ShouldEqual, 13)

	img, err := openImage("inputs/white_bg.png")
	test.That(t, err, test.ShouldBeNil)
	test.That(t, img.Bounds().Empty(), test.ShouldBeFalse)

	t.ResetTimer()

	for t.Loop() {
		imgMatrix := ImageToMatrix(img, scale)
		detections := findTriangles(templates, imgMatrix, 2, .65, scale)
		test.That(t, len(detections), test.ShouldEqual, 3)
	}

}

// TestTemplateResizing tests that templates are resized proportionally
func TestTemplateResizing(t *testing.T) {
	// Use an actual template image
	img, err := openImage("templates/triangle_1.png")
	test.That(t, err, test.ShouldBeNil)

	originalSize := img.Bounds().Size()
	t.Logf("Original template size: %v", originalSize)

	// Create scaled template image
	template, err := NewTemplateFromImage(img, 0.8)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, template.originalSize, test.ShouldResemble, originalSize)

	// Verify kernel dimensions are scaled correctly
	expectedWidth := int(float64(originalSize.X) * 0.8)
	expectedHeight := int(float64(originalSize.Y) * 0.8)
	test.That(t, template.kernelWidth, test.ShouldEqual, expectedWidth)
	test.That(t, template.kernelHeight, test.ShouldEqual, expectedHeight)
}

// tests that detected coordinates are properly scaled
func TestCoordinateScaling(t *testing.T) {
	templates, err := loadTemplates(0.5)
	test.That(t, err, test.ShouldBeNil)

	img, err := openImage("inputs/input_1.jpg")
	test.That(t, err, test.ShouldBeNil)
	originalSize := img.Bounds().Size()
	t.Logf("Original input image size: %v", originalSize)
	// Process image
	matrix := ImageToMatrix(img, 0.5)
	t.Logf("Resized input image size: %dx%d", len(matrix[0]), len(matrix))
	detections := findTriangles(templates, matrix, 2, 0.65, 0.5)

	for i, det := range detections {
		box := det.BoundingBox()
		t.Logf("Detection %d: Box=%v, Score=%f", i, box, det.Score())

		// Verify coordinates are in original image space
		test.That(t, box.Min.X, test.ShouldBeLessThan, originalSize.X)
		test.That(t, box.Min.Y, test.ShouldBeLessThan, originalSize.Y)
		test.That(t, box.Max.X, test.ShouldBeLessThan, originalSize.X)
		test.That(t, box.Max.Y, test.ShouldBeLessThan, originalSize.Y)

		// verify the box size matches one of our templates
		boxWidth := box.Dx()
		boxHeight := box.Dy()

		// check if this detection matches any template size
		foundMatchingTemplate := false
		for _, template := range templates {
			if boxWidth == template.originalSize.X && boxHeight == template.originalSize.Y {
				foundMatchingTemplate = true
				break
			}
		}
		test.That(t, foundMatchingTemplate, test.ShouldBeTrue)
	}
}

// plot histogram of correlation values to see distibution for finding best threshold
func TestCorrelationHistogramGonum(t *testing.T) {
	scale := 0.5
	templates, err := loadTemplates(scale)
	test.That(t, err, test.ShouldBeNil)

	img, err := openImage("inputs/image_2.png")
	test.That(t, err, test.ShouldBeNil)

	imgMatrix := ImageToMatrix(img, scale)

	// all correlation values from all templates
	var allCorrelations []float32
	lowThreshold := float32(0.4)

	for _, template := range templates {
		matches := template.FindMatch(imgMatrix, 2, lowThreshold, scale)
		for _, match := range matches {
			allCorrelations = append(allCorrelations, match.Score)
		}
	}

	t.Logf("Total correlation values collected: %d", len(allCorrelations))

	p := plot.New()
	p.Title.Text = "Correlation Values Distribution"
	p.X.Label.Text = "Correlation"
	p.Y.Label.Text = "Frequency"

	// Convert correlations to plotter.Values
	values := make(plotter.Values, len(allCorrelations))
	for i, corr := range allCorrelations {
		values[i] = float64(corr)
	}

	// plot histogram
	hist, err := plotter.NewHist(values, 50)
	if err != nil {
		t.Fatal(err)
	}

	hist.FillColor = color.RGBA{100, 150, 255, 255}
	p.Add(hist)

	// save plot
	if err := p.Save(8*vg.Inch, 4*vg.Inch, "detections/correlation_histogram.png"); err != nil {
		t.Fatal(err)
	}
}
