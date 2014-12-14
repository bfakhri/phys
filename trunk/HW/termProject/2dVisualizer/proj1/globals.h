#ifndef GLOBALS_H
#define GLOBALS_H

#include "all_includes.h"

#define GAME_RESET 0
#define GAME_QUIT 1
#define GAME_FILLED 2
#define GAME_WIRE 3

// Time variables
extern float ext_refreshRate;			// Times per seconds the window is refreshed (ideally)
extern float ext_clocksPerRefresh;		// Clocks elapsed before next refresh
extern clock_t ext_lastRefreshClock;	// Last time (in ticks) that screen was refreshed
extern unsigned int ext_secGameTime;	// Time in seconds since game started
extern unsigned int ext_frameCount;		// Frames since last second in game

// Window variables
extern unsigned int ext_winWidth;			// Holds width of window
extern unsigned int ext_winHeight;			// Holds height of window
extern unsigned int ext_winHeightOffset;	// Holds offset to center the viewports vertically
extern unsigned int ext_winWidthOffset;		// Holds offset to center the viewports horizontally

// Scoring
extern unsigned int ext_score;				// keeps score of game
extern unsigned int ext_wallUniTargets;		// Number of wall targets in play
extern unsigned int ext_leftUniTargets;		// Number of left uni targets in play
extern unsigned int ext_rightUniTargets;	// Number of right uni targets in play
extern unsigned int ext_targetsLeft;		// Number of total targets left in play

// Shooting
extern float ext_missileSpeed;		// Defualt missile speed in units/frame

// Graphics 
extern bool ext_filled;				// Whether or not the shapes are filled

#endif
