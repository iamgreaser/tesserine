/*
	tesserine: 4-dimensional Minecraft-like game
	Copyright (C) 2013 Ben "GreaseMonkey" Russell
	
	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
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

typedef union v4f
{
	float a[4]; // array
	int i[4]; // integer array

	struct { float x,y,z,w; } v; // vertex/vector
	struct { float s,t,r,q; } t; // texture
	struct { float r,g,b,a; } c; // colour

	__m128 m; // __m128
} v4f_t;

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
	int x1 = (x0 + d) & (size - 1);
	int z1 = (z0 + d) & (size - 1);
	int w1 = (w0 + d) & (size - 1);

	int xc = ((x0 + x1)>>1) & (size - 1);
	int zc = ((z0 + z1)>>1) & (size - 1);
	int wc = ((w0 + w1)>>1) & (size - 1);

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
	wl->d.v.x = size;
	wl->d.v.y = size;
	wl->d.v.z = size;
	wl->d.v.w = size;

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

static void render_surface_trace(uint32_t *pixel, int length, __m128i ci, __m128 cf, __m128 bt0, __m128 bt1, __m128 t0, __m128 t1, __m128i g)
{
	//
}

static void render_span_split_sections(int sec, uint32_t *pixel, int length, __m128i ci, __m128 cf, const v4f_t *s0, const v4f_t *s1, const v4f_t *v0, const v4f_t *v1)
{
	if(length == 0)
		return;

	if(sec == 4)
	{
		// get abs velocities
		__m128 av0 = _mm_and_ps((__m128)_mm_set1_epi32(0x7FFFFFFF), v0->m);
		__m128 av1 = _mm_and_ps((__m128)_mm_set1_epi32(0x7FFFFFFF), v1->m);

		// get base times
		__m128 bt0 = _mm_rcp_ps(av0);
		__m128 bt1 = _mm_rcp_ps(av1);

		// generate pixel data
		// TODO: move this through the chain a bit...
		v4f_t v;
		v.m = v0->m;
		__m128i cpack;
		
		//cpack = _mm_cvtps_epi32(_mm_add_ps(_mm_set1_ps(127.5f), _mm_mul_ps(_mm_set1_ps(255.0f/2.0f), v.m)));
		uint32_t *cbase = s0->a;
		uint32_t cval = 
			((cbase[0]&0xFF)<<0)^
			((cbase[1]&0xFF)<<8)^
			((cbase[2]&0xFF)<<16)^
			(cbase[3]&0xAAAAAAAA);

		int i;
		for(i = 0; i < length; i++)
			*(pixel++) = cval;
	} else {
		// check for a split
		if(s0->i[sec] != s1->i[sec])
		{
			// there's a split. interpolate the velocities to find the centre.
			float t0 = v0->a[sec];
			float t1 = v1->a[sec];
			float td = (t1-t0);
			if(td < EPSILON && td >= 0.0f) td = EPSILON;
			else if(td < 0.0f && td >= -EPSILON) td = -EPSILON;
			float t = -t0/td;
			if(t >= 1.0f) t = 1.0f;
			if(t <= 0.0f) t = 0.0f;
			float at = fabsf(t);

			v4f_t vc;
			vc.m = _mm_add_ps(v0->m,
				_mm_mul_ps(
					_mm_sub_ps(v1->m, v0->m),
					_mm_set1_ps(t)));

			// set up our signs correctly
			v4f_t sn0, sn1;
			sn0.m = s1->m;
			sn1.m = s0->m;
			sn0.i[sec] = s0->i[sec];
			sn1.i[sec] = s1->i[sec];

			int lenoffs = length*at;

			// split along.
			render_span_split_sections(sec + 1, pixel, lenoffs, ci, cf, s0, &sn0, v0, &vc);
			render_span_split_sections(sec + 1, pixel + lenoffs, length - lenoffs, ci, cf, &sn1, s1, &vc, v1);
		} else {
			// no split. move onto the next section.
			render_span_split_sections(sec + 1, pixel, length, ci, cf, s0, s1, v0, v1);
		}
	}
}

static void render_span(const camera_t *c, uint32_t *pixel, int length, float x0, float x1, float y)
{
	// apply camera matrix
	v4f_t p, v0, v1, v;
	p.m = c->p.m;
	v0.m = _mm_mul_ps(_mm_set1_ps(x0), c->o[0].m);
	v0.m = _mm_add_ps(v0.m, _mm_mul_ps(_mm_set1_ps(y), c->o[1].m));
	v0.m = _mm_add_ps(v0.m, _mm_mul_ps(_mm_set1_ps(1.0f), c->o[2].m));
	v1.m = _mm_mul_ps(_mm_set1_ps(x1), c->o[0].m);
	v1.m = _mm_add_ps(v1.m, _mm_mul_ps(_mm_set1_ps(y), c->o[1].m));
	v1.m = _mm_add_ps(v1.m, _mm_mul_ps(_mm_set1_ps(1.0f), c->o[2].m));

	// calculate signs
	v4f_t s0, s1;
	s0.m = (__m128)_mm_cmplt_ps(v0.m, _mm_setzero_ps());
	s1.m = (__m128)_mm_cmplt_ps(v1.m, _mm_setzero_ps());

	// calculate current cell int/frac parts
	__m128i ci = _mm_cvtps_epi32(p.m);
	__m128 cf = _mm_cvtepi32_ps(ci);

	// find the velocity splits
	render_span_split_sections(0, pixel, length, ci, cf, &s0, &s1, &v0, &v1);
}

void render_screen()
{
	int x,y;

	for(y = 0; y < screen->h; y++)
	{
		// calculate
		uint32_t *p = (uint32_t *)(screen->pixels + (screen->pitch * y));
		float fy = ((y*2.0f-screen->h)/screen->h);
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
	}
}

int main(int argc, char *argv[])
{
	printf("tesserine Copyright (C) 2013 Ben \"GreaseMonkey\" Russell\n");
	SDL_Init(SDL_INIT_TIMER | SDL_INIT_VIDEO);

	signal(SIGINT, SIG_DFL);
	signal(SIGTERM, SIG_DFL);

	SDL_WM_SetCaption("tesserine prealpha", NULL);
	screen = SDL_SetVideoMode(800, 600, 32, 0);

	init_camera(&cam);

	int quitflag = 0;

	bworld = world_gen(32); // you shouldn't have to go higher... 128 is OK (256MB of RAM) but not 256 (2GB of RAM, I think)

	SDL_WarpMouse(screen->w/2, screen->h/2);
	int inhibit_warp = 1;

	while(!quitflag)
	{
		render_screen();
		SDL_Flip(screen);
		SDL_Delay(10);

		SDL_Event ev;

		while(SDL_PollEvent(&ev))
		switch(ev.type)
		{
			case SDL_QUIT:
				quitflag = 1;
				break;
			case SDL_MOUSEMOTION:
				if(inhibit_warp)
					inhibit_warp = 0;
				else
				{
					cam_rotate(&cam, ev.motion.xrel, ev.motion.yrel, 0.0f);
					SDL_WarpMouse(screen->w/2, screen->h/2);
					inhibit_warp = 1;
					while(SDL_PollEvent(&ev));
				}
				break;
			case SDL_MOUSEBUTTONDOWN:
				if(ev.button.button == 4)
					cam_rotate(&cam, 0.0f, 0.0f, 1.0f);
				else if(ev.button.button == 5)
					cam_rotate(&cam, 0.0f, 0.0f, -1.0f);
				break;
		}
	}

	return 0;
}

