#include "capture.h"

Recorder::Recorder()
{
	frameCount = 0; 
	time = 0; 
}

void Recorder::capture(float period)
{
	time += period; 
	if(time >= 1/double(VIDEO_FPS)){
		time -= 1/double(VIDEO_FPS); 
		frameCount++;
	}else{
		return;
	}

	uint8_t ** img_buffer = new uint8_t*[SCREEN_RES_X];
	for(int y=0; y<SCREEN_RES_Y; y++){
		img_buffer[y] = new uint8_t[SCREEN_RES_X*3];
		for(int i=0; i<SCREEN_RES_X*3; i++){ 
			img_buffer[y][i] = 0xFF; 
		}
	}

	glPixelStorei(GL_PACK_ALIGNMENT, 1);

	//glReadBuffer(GL_FRONT);
	// Get image from buffer
	for(int y=0; y<SCREEN_RES_Y; y++){
		for(int x=0; x<SCREEN_RES_X; x++){
			//glReadPixels(x, y, SCREEN_RES_X, SCREEN_RES_Y, GL_RGB, GL_UNSIGNED_BYTE, &img_buffer[SCREEN_RES_Y-y-1][x*3]);
			glReadPixels(x, y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, &img_buffer[SCREEN_RES_Y-y-1][x*3]);

			GLenum error; 
			if ((error = glGetError()) != GL_NO_ERROR)
				printf("GL Error: %s (%s)\n", gluErrorString(error));
			//img_buffer[y][x+0] = 0;
			//img_buffer[y][x+1] = 0;
			//img_buffer[y][x+2] = 0;
		}
	}

	GLenum error; 
	if ((error = glGetError()) != GL_NO_ERROR)
		printf("GL Error: %s (%s)\n", gluErrorString(error));

	char filename[40];
	sprintf (filename, "./images/%d.png", frameCount);	
	
	// Write out image to PNG file
	writeImage(filename, SCREEN_RES_X, SCREEN_RES_Y, img_buffer, "title"); 
}

// --- Ripped From --- 
// LibPNG example
// A.Greensted
// http://www.labbookpages.co.uk
int writeImage(char* filename, int width, int height, uint8_t** buffer, char* title)
 {
	int code = 0;
	FILE *fp = NULL;
	png_structp png_ptr = NULL;
	png_infop info_ptr = NULL;
	png_bytep row = NULL;
	
	// Open file for writing (binary mode)
	fp = fopen(filename, "wb");
	if (fp == NULL) {
		fprintf(stderr, "Could not open file %s for writing\n", filename);
		code = 1;
		goto finalise;
	}

	// Initialize write structure
	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (png_ptr == NULL) {
		fprintf(stderr, "Could not allocate write struct\n");
		code = 1;
		goto finalise;
	}

	// Initialize info structure
	info_ptr = png_create_info_struct(png_ptr);
	if (info_ptr == NULL) {
		fprintf(stderr, "Could not allocate info struct\n");
		code = 1;
		goto finalise;
	}

	// Setup Exception handling
	if (setjmp(png_jmpbuf(png_ptr))) {
		fprintf(stderr, "Error during png creation\n");
		code = 1;
		goto finalise;
	}

	png_init_io(png_ptr, fp);

	// Write header (8 bit colour depth)
	png_set_IHDR(png_ptr, info_ptr, width, height,
			8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
			PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	// Set title
	if (title != NULL) {
		png_text title_text;
		title_text.compression = PNG_TEXT_COMPRESSION_NONE;
		title_text.key = "Title";
		title_text.text = title;
		png_set_text(png_ptr, info_ptr, &title_text, 1);
	}

	png_write_info(png_ptr, info_ptr);

	// Write image data
	int x, y;
	for (y=0 ; y<height ; y++) {
		png_write_row(png_ptr, buffer[y]);
	}

	// End write
	png_write_end(png_ptr, NULL);

	finalise:
	if (fp != NULL) fclose(fp);
	if (info_ptr != NULL) png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
	if (png_ptr != NULL) png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
	if (row != NULL) free(row);

	return code;
}
