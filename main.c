/*
	tesserine: 4-dimensional Minecraft-like game
	Copyright (C) 2013 Ben "GreaseMonkey" Russell
	
	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 2 of the License, or
	(at your option) any later version.
	
	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.
	
	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <errno.h>

#include <math.h>

#include <SDL.h>

#include <signal.h>

// MMX, SSE, SSE2 respectively
#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>

#define EPSILON 0.00000000000001f

static uint32_t facecol[4] = {
	0xFF0000FF,
	0xFF00FF00,
	0xFFFF0000,
	0xFFCC7700,
};

typedef union v4f
{
	float a[4]; // array
	int i[4]; // integer array

	struct { int x,y,z,w; } vi; // vertex/vector

	struct { float x,y,z,w; } v; // vertex/vector
	struct { float s,t,r,q; } t; // texture
	struct { float r,g,b,a; } c; // colour

	__m128 m; // __m128
	__m128i mi; // __m128i
} v4f_t;

typedef union
{
	float f;
	int i;
} flint;

typedef struct camera
{
	v4f_t p; // position
	v4f_t o[4]; // orientation matrix
} camera_t;

typedef struct world
{
	v4f_t d; // dimensions
	uint8_t data[];
} world_t;

SDL_Surface *screen;
camera_t cam;
world_t *bworld = NULL;
int fps_counter = 0;
int fps_waituntil = 0;

float ufrand(void)
{
	return (rand() % 65539) / 65539.0f;
}

float sfrand(void)
{
	return (ufrand() * 2.0f) - 1.0f;
}

void world_hmap_gen(int size, float *hmap, int x0, int z0, int w0, float amp, int d)
{
	// cancel if d == 1
	if(d <= 1)
		return;

	// get coordinates
	int x1b = (x0 + d);
	int z1b = (z0 + d);
	int w1b = (w0 + d);

	int x1 = x1 & (size - 1);
	int z1 = z1 & (size - 1);
	int w1 = w1 & (size - 1);

	int xc = ((x0 + x1b)>>1) & (size - 1);
	int zc = ((z0 + z1b)>>1) & (size - 1);
	int wc = ((w0 + w1b)>>1) & (size - 1);

	// collect points
	float v000 = hmap[x0 + size*(z0 + size*w0)];
	float v001 = hmap[x1 + size*(z0 + size*w0)];
	float v010 = hmap[x0 + size*(z1 + size*w0)];
	float v011 = hmap[x1 + size*(z1 + size*w0)];
	float v100 = hmap[x0 + size*(z0 + size*w1)];
	float v101 = hmap[x1 + size*(z0 + size*w1)];
	float v110 = hmap[x0 + size*(z1 + size*w1)];
	float v111 = hmap[x1 + size*(z1 + size*w1)];

	// calc line averages
	float v00c = (v000 + v001) / 2.0f + amp*sfrand();
	float v01c = (v010 + v011) / 2.0f + amp*sfrand();
	float v10c = (v100 + v101) / 2.0f + amp*sfrand();
	float v11c = (v110 + v111) / 2.0f + amp*sfrand();
	float v0c0 = (v000 + v010) / 2.0f + amp*sfrand();
	float v1c0 = (v100 + v110) / 2.0f + amp*sfrand();
	float v0c1 = (v001 + v011) / 2.0f + amp*sfrand();
	float v1c1 = (v101 + v111) / 2.0f + amp*sfrand();
	float vc00 = (v000 + v100) / 2.0f + amp*sfrand();
	float vc01 = (v001 + v101) / 2.0f + amp*sfrand();
	float vc10 = (v010 + v110) / 2.0f + amp*sfrand();
	float vc11 = (v011 + v111) / 2.0f + amp*sfrand();

	amp *= 0.9f;

	// calc plane averages
	float v0cc = (v00c + v01c) / 2.0f + amp*sfrand();
	float v1cc = (v10c + v11c) / 2.0f + amp*sfrand();
	float vcc0 = (v0c0 + v1c0) / 2.0f + amp*sfrand();
	float vcc1 = (v0c1 + v1c1) / 2.0f + amp*sfrand();
	float vc0c = (vc00 + vc01) / 2.0f + amp*sfrand();
	float vc1c = (vc10 + vc11) / 2.0f + amp*sfrand();

	amp *= 0.9f;

	// calc volume average
	float vccc = (v0cc + v1cc) / 2.0f + amp*sfrand();

	amp *= 0.9f;

	// store averages
	hmap[x0 + size*(z0 + size*w0)] = v000;
	hmap[x1 + size*(z0 + size*w0)] = v001;
	hmap[xc + size*(z0 + size*w0)] = v00c;
	hmap[x0 + size*(z1 + size*w0)] = v010;
	hmap[x1 + size*(z1 + size*w0)] = v011;
	hmap[xc + size*(z1 + size*w0)] = v01c;
	hmap[x0 + size*(zc + size*w0)] = v0c0;
	hmap[x1 + size*(zc + size*w0)] = v0c1;
	hmap[xc + size*(zc + size*w0)] = v0cc;
	hmap[x0 + size*(z0 + size*w1)] = v100;
	hmap[x1 + size*(z0 + size*w1)] = v101;
	hmap[xc + size*(z0 + size*w1)] = v10c;
	hmap[x0 + size*(z1 + size*w1)] = v110;
	hmap[x1 + size*(z1 + size*w1)] = v111;
	hmap[xc + size*(z1 + size*w1)] = v11c;
	hmap[x0 + size*(zc + size*w1)] = v1c0;
	hmap[x1 + size*(zc + size*w1)] = v1c1;
	hmap[xc + size*(zc + size*w1)] = v1cc;
	hmap[x0 + size*(z0 + size*wc)] = vc00;
	hmap[x1 + size*(z0 + size*wc)] = vc01;
	hmap[xc + size*(z0 + size*wc)] = vc0c;
	hmap[x0 + size*(z1 + size*wc)] = vc10;
	hmap[x1 + size*(z1 + size*wc)] = vc11;
	hmap[xc + size*(z1 + size*wc)] = vc1c;
	hmap[x0 + size*(zc + size*wc)] = vcc0;
	hmap[x1 + size*(zc + size*wc)] = vcc1;
	hmap[xc + size*(zc + size*wc)] = vccc;

	// recurse
	int nd = d>>1;
	world_hmap_gen(size, hmap, x0, z0, w0, amp, nd);
	world_hmap_gen(size, hmap, xc, z0, w0, amp, nd);
	world_hmap_gen(size, hmap, x0, zc, w0, amp, nd);
	world_hmap_gen(size, hmap, xc, zc, w0, amp, nd);
	world_hmap_gen(size, hmap, x0, z0, wc, amp, nd);
	world_hmap_gen(size, hmap, xc, z0, wc, amp, nd);
	world_hmap_gen(size, hmap, x0, zc, wc, amp, nd);
	world_hmap_gen(size, hmap, xc, zc, wc, amp, nd);
}

world_t *world_gen(int size)
{
	int x,y,z,w;

	printf("Generating world...");
	fflush(stdout);

	// create world object
	world_t *wl = malloc(sizeof(world_t) + size*size*size*size);
	wl->d.vi.x = size;
	wl->d.vi.y = size;
	wl->d.vi.z = size;
	wl->d.vi.w = size;

	// create heightmap
	float *hmap = malloc(sizeof(float) * size*size*size);
	hmap[0] = sfrand();
	world_hmap_gen(size, hmap, 0, 0, 0, 1.0f, size);

	// fill the world with blocks
	for(x = 0; x < size; x++)
	for(z = 0; z < size; z++)
	for(w = 0; w < size; w++)
	{
		float hv = hmap[x + size*(z + size*w)];
		int height = size * (0.5f*sinf(hv) + 0.5f) + 0.5f;

		for(y = 0; y < size; y++)
			wl->data[y + size*(x + size*(z + size*w))] = (y < height ? 1 : 0);
	}

	// free heightmap
	free(hmap);

	// DONE
	printf(" Done!\n");

	return wl;
}

void rotate_pair(float amt, v4f_t *a, v4f_t *b)
{
	amt = M_PI/10.0f * amt;

	__m128 va = a->m;
	__m128 vb = b->m;

	float s = sinf(amt);
	float c = cosf(amt);

	a->m = _mm_add_ps(
		_mm_mul_ps(_mm_set1_ps(c), va),
		_mm_mul_ps(_mm_set1_ps(-s), vb));
	b->m = _mm_add_ps(
		_mm_mul_ps(_mm_set1_ps(s), va),
		_mm_mul_ps(_mm_set1_ps(c), vb));
}

void cam_rotate(camera_t *c, float x, float y, float w)
{
	rotate_pair(x/100.0f, &(c->o[0]), &(c->o[2]));
	rotate_pair(y/100.0f, &(c->o[1]), &(c->o[2]));
	rotate_pair(w/3.0f, &(c->o[3]), &(c->o[2]));

	// This is the X roll correction.
	rotate_pair(asinf(c->o[0].v.y), &(c->o[0]), &(c->o[1]));

	// Hmm. If we use this it MIGHT be useful. Or just awful. Who knows.
	// Anyway, it's W roll correction, just in case we need it.
	//rotate_pair(asinf(c->o[3].v.y), &(c->o[3]), &(c->o[1]));
}

void init_camera(camera_t *c)
{
	// note, these are loaded w, z, y, x!
	c->p.m = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f);

	c->o[0].m = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f);
	c->o[1].m = _mm_set_ps(0.0f, 0.0f, 1.0f, 0.0f);
	c->o[2].m = _mm_set_ps(0.0f, 1.0f, 0.0f, 0.0f);
	c->o[3].m = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
}

static uint32_t render_pixel(const v4f_t *ip, const v4f_t *iv)
{
	int i;

	__m128 p = ip->m;
	__m128 v = iv->m;

	// get sign
	v4f_t v_isneg;
	v_isneg.m = _mm_cmplt_ps(v, _mm_setzero_ps());

	// get distance to boundary
	v4f_t cell, db;
	cell.mi = _mm_cvtps_epi32(p);
	__m128 db0 = _mm_sub_ps(p, _mm_cvtepi32_ps(cell.mi));
	__m128 db1 = _mm_sub_ps(_mm_set1_ps(1.0f), db0);
	__m128 dbd = _mm_sub_ps(db0, db1);
	db.m = _mm_add_ps(db1, _mm_and_ps(v_isneg.m, dbd));

	// get positive velocity
	__m128 vpos = (__m128)_mm_and_si128(
		_mm_set1_epi32(0x7FFFFFFF), (__m128i)v);
	vpos = _mm_max_ps(_mm_set1_ps(EPSILON), vpos);

	const static uint32_t faces[4] = {
		0xFF0000FF,
		0xFF00FF00,
		0xFFFF0000,
		0xFFAA00AA,
	};

	// get world dims
	v4f_t ws;
	ws.mi = bworld->d.mi;

	// trace!
	int face = 0;
	for(i = 0; i < 30; i++)
	{
		// calculate time to face
		v4f_t t;
		t.m = (__m128)_mm_or_si128(
			_mm_setr_epi32(0,1,2,3),
			_mm_and_si128((__m128i)_mm_div_ps(db.m, vpos),
				_mm_set1_epi32(~3)));
		t.m = _mm_min_ps(_mm_movehl_ps(t.m, t.m), _mm_movelh_ps(t.m, t.m));

		float tf = (t.a[0] < t.a[1] ? t.a[0] : t.a[1]);
		face = (*(int *)&tf) & 3;

		db.m = _mm_sub_ps(db.m,
			_mm_mul_ps(_mm_set1_ps(tf), vpos));
		cell.i[face] += (v_isneg.i[face] ? -1 : 1);
		db.a[face] += 1.0f;

		if(cell.i[face] < 0 || cell.i[face] >= ws.i[face])
			return 0xFF00AAFF;

		int idx = cell.vi.y + ws.vi.y*(
			cell.vi.x + ws.vi.x*(
			cell.vi.z + ws.vi.z*(
			cell.vi.w)));

		if(bworld->data[idx])
			return faces[face];
	}

	return 0xFFFFFFFF;
}

static void render_span(const camera_t *c, uint32_t *pixel, int length, float x0, float x1, float y)
{
	// apply camera matrix
	v4f_t p, vb, vi, v;
	p.m = c->p.m;
	vb.m = _mm_mul_ps(_mm_set1_ps(x0), c->o[0].m);
	vb.m = _mm_add_ps(vb.m, _mm_mul_ps(_mm_set1_ps(y), c->o[1].m));
	vb.m = _mm_add_ps(vb.m, _mm_mul_ps(_mm_set1_ps(1.0f), c->o[2].m));
	vi.m = _mm_mul_ps(_mm_set1_ps(3.0f*(x1-x0)/(float)length), c->o[0].m);

	// calculate current cell int/frac parts
	__m128i ci = _mm_cvtps_epi32(p.m);
	__m128 cf = _mm_sub_ps(p.m, _mm_cvtepi32_ps(ci));

	// render each pixel
	int sx;
	for(sx = 0; sx < length; sx+=3)
	{
		pixel[0] = pixel[1] = pixel[2] = render_pixel(&p, &vb);
		pixel += 3;
		vb.m = _mm_add_ps(vb.m, vi.m);
	//render_span_split_sections(0, pixel, length, ci, cf, &s0, &s1, &v0, &v1);
	}
}

void render_screen()
{
	int x,y;

#pragma omp parallel for
	for(y = 0; y < screen->h; y+=3)
	{
		// calculate
		uint32_t *p = (uint32_t *)(screen->pixels + (screen->pitch * y));
		float fy = -((y*2.0f-screen->h)/screen->h);
		float fx0 = -1.0f;
		float fx1 = 1.0f;

		// apply projection
		if(screen->h < screen->w)
			fy *= ((float)screen->h)/(float)screen->w;
		else {
			fx0 *= ((float)screen->w)/(float)screen->h;
			fx1 *= ((float)screen->w)/(float)screen->h;
		}

		// render
		render_span(&cam, p, screen->w, fx0, fx1, fy);
		memcpy((screen->pitch + (uint8_t *)p), p, screen->w*4);
		memcpy((2*screen->pitch + (uint8_t *)p), p, screen->w*4);
	}
}

int main(int argc, char *argv[])
{
	printf("tesserine Copyright (C) 2013 Ben \"GreaseMonkey\" Russell\n");
	SDL_Init(SDL_INIT_TIMER | SDL_INIT_VIDEO);

	signal(SIGINT, SIG_DFL);
	signal(SIGTERM, SIG_DFL);

	SDL_WM_SetCaption("tesserine prealpha", NULL);
	screen = SDL_SetVideoMode(960, 600, 32, 0);

	int quitflag = 0;

	_MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);

	bworld = world_gen(32); // you shouldn't have to go higher... 128 is OK (256MB of RAM) but not 256 (2GB of RAM, I think)
	init_camera(&cam);
	cam.p.v.x = (bworld->d.vi.x+1.0f)/2.0f;
	cam.p.v.y = (bworld->d.vi.y+1.0f)/2.0f;
	cam.p.v.z = (bworld->d.vi.z+1.0f)/2.0f;
	cam.p.v.w = (bworld->d.vi.w+1.0f)/2.0f;

	v4f_t cv;
	cv.m = _mm_set1_ps(0.0f);

	SDL_WarpMouse(screen->w/2, screen->h/2);
	int inhibit_warp = 1;
	fps_waituntil = SDL_GetTicks() + 100;

	while(!quitflag)
	{
		cam.p.m = _mm_add_ps(cam.p.m,
			_mm_mul_ps(cam.o[2].m,
				_mm_set1_ps(0.05f * cv.v.z)));
		cam.p.m = _mm_add_ps(cam.p.m,
			_mm_mul_ps(cam.o[0].m,
				_mm_set1_ps(0.05f * cv.v.x)));
		render_screen();
		SDL_Flip(screen);

		fps_counter++;
		int ntime = SDL_GetTicks();
		if(ntime >= fps_waituntil)
		{
			char buf[80];
			sprintf(buf, "tesserine | FPS: %i", fps_counter);
			fps_counter = 0;
			SDL_WM_SetCaption(buf, NULL);
			fps_waituntil += 1000;
		}

		//SDL_Delay(10);

		SDL_Event ev;

		while(SDL_PollEvent(&ev))
		switch(ev.type)
		{
			case SDL_QUIT:
				quitflag = 1;
				break;
			case SDL_KEYDOWN: switch(ev.key.keysym.sym)
			{
				case SDLK_w: cv.v.z =  1.0f; break;
				case SDLK_s: cv.v.z = -1.0f; break;
				case SDLK_a: cv.v.x = -1.0f; break;
				case SDLK_d: cv.v.x =  1.0f; break;
			} break;
			case SDL_KEYUP: switch(ev.key.keysym.sym)
			{
				case SDLK_w: cv.v.z =  0.0f; break;
				case SDLK_s: cv.v.z =  0.0f; break;
				case SDLK_a: cv.v.x =  0.0f; break;
				case SDLK_d: cv.v.x =  0.0f; break;
			} break;

			case SDL_MOUSEMOTION:
				if(!inhibit_warp)
				{
					cam_rotate(&cam, ev.motion.xrel, -ev.motion.yrel, 0.0f);
					SDL_WarpMouse(screen->w/2, screen->h/2);
					//while(SDL_PollEvent(&ev));
					inhibit_warp = 1;
				}
				break;
			case SDL_MOUSEBUTTONDOWN:
				if(ev.button.button == 4)
					cam_rotate(&cam, 0.0f, 0.0f, 1.0f);
				else if(ev.button.button == 5)
					cam_rotate(&cam, 0.0f, 0.0f, -1.0f);
				break;
		}

		inhibit_warp = 0;
	}

	return 0;
}

