package triangle_on_sonar_finder

import (
	"embed"
	"fmt"
	"image"
	"image/color"
	"path/filepath"
	"sort"
	"strings"

	objdet "go.viam.com/rdk/vision/objectdetection"
)

//go:embed templates/*
var templateFS embed.FS

// loadTemplates loads template images from the specified directory and returns
// a slice of TemplateFromImage objects. Each template is normalized. Returns an error if the directory cannot be accessed or if
// no valid templates are found.
func loadTemplates(scale float64) ([]TemplateFromImage, error) {
	validExtensions := []string{".png", ".jpg", ".jpeg"}

	files, err := templateFS.ReadDir("templates")
	if err != nil {
		return nil, fmt.Errorf("error reading template directory: %v", err)
	}

	templates := []TemplateFromImage{}

	for _, file := range files {
		if file.IsDir() {
			continue
		}

		filename := file.Name()
		isValidExt := false
		for _, ext := range validExtensions {
			if strings.HasSuffix(strings.ToLower(filename), ext) {
				isValidExt = true
				break
			}
		}

		if !isValidExt {
			continue
		}

		f, err := templateFS.Open(filepath.Join("templates", filename))
		if err != nil {
			return nil, fmt.Errorf("cannot open file [%s]: %w", filename, err)
		}
		defer f.Close()

		img, _, err := image.Decode(f)
		if err != nil {
			return nil, fmt.Errorf("error decoding image (%s): %v", filename, err)
		}

		template, err := NewTemplateFromImage(img, scale)
		if err != nil {
			return nil, fmt.Errorf("cannot create template from [%s]: %w", filename, err)
		}
		templates = append(templates, *template)
	}
	return templates, nil
}

// ImageToMatrix converts a grayscale image to a 2D float32 matrix -- preprocessing image using sobel edge detection and resizing
func ImageToMatrix(img image.Image, scale float64) [][]float64 {
	originalWidth := img.Bounds().Dx()
	// step 1: resize image
	img = resizeImage(img, uint(float64(originalWidth)*scale)) //resizing image
	bounds := img.Bounds()
	newWidth := bounds.Dx()
	newHeight := bounds.Dy()

	// step 2: convert to grayscale matrix (same logic for template)
	grayMatrix := make([][]float64, newHeight)
	for y := 0; y < newHeight; y++ {
		grayMatrix[y] = make([]float64, newWidth)
		for x := 0; x < newWidth; x++ {
			c := img.At(x+bounds.Min.X, y+bounds.Min.Y)
			grayValue := float64(color.GrayModel.Convert(c).(color.Gray).Y)
			grayMatrix[y][x] = grayValue
		}
	}

	// step 3: apply Sobel edge detection
	edgeMatrix := sobelEdge(grayMatrix, newWidth, newHeight, 50) // adjust threshold as needed
	// step 4: return the edge matrix [][]float64
	return edgeMatrix
}

// calculateIoU calculates the Intersection over Union between two rectangles
func calculateIoU(box1, box2 *image.Rectangle) float64 {
	if box1 == nil || box2 == nil {
		return 0
	}

	// Calculate intersection rectangle
	intersection := box1.Intersect(*box2)
	if intersection.Empty() {
		return 0
	}

	// Calculate areas
	intersectionArea := intersection.Dx() * intersection.Dy()
	box1Area := box1.Dx() * box1.Dy()
	box2Area := box2.Dx() * box2.Dy()

	// Calculate IoU
	unionArea := box1Area + box2Area - intersectionArea
	return float64(intersectionArea) / float64(unionArea)
}

func findTriangles(templates []TemplateFromImage, imgMatrix [][]float64, stride int, threshold float32, scale float64) []objdet.Detection {
	// Find matches using all templates
	var allMatches []Match
	for _, template := range templates {
		matches := template.FindMatch(imgMatrix, stride, threshold, scale)
		allMatches = append(allMatches, matches...)
	}

	// Convert matches to detections
	detections := make([]objdet.Detection, 0, len(allMatches))
	for _, match := range allMatches {
		box := match.GetBoundingBox()
		det := objdet.NewDetectionWithoutImgBounds(box, float64(match.Score), "triangle")
		detections = append(detections, det)
	}

	// Sort detections by score in descending order
	sort.Slice(detections, func(i, j int) bool {
		return detections[i].Score() > detections[j].Score()
	})

	// Apply Non-Maximum Suppression
	var filteredDetections []objdet.Detection
	used := make([]bool, len(detections))

	for i := 0; i < len(detections); i++ {
		if used[i] {
			continue
		}

		// Keep the current detection
		filteredDetections = append(filteredDetections, detections[i])
		used[i] = true

		// Check overlap with remaining detections
		for j := i + 1; j < len(detections); j++ {
			if used[j] {
				continue
			}

			// Calculate IoU between current and remaining detection
			iou := calculateIoU(detections[i].BoundingBox(), detections[j].BoundingBox())

			// If IoU is greater than threshold, mark as used
			if iou > 0.3 { // You can adjust this threshold
				used[j] = true
			}
		}
	}

	return filteredDetections
}
