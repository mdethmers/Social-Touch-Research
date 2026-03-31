$fn = 128;
include <BOSL2/std.scad>
include <BOSL2/beziers.scad>

// 350 and 200 diameters
// total height 1166.66

// total side length = sqrt(1166.66^2 + 175^2) = 1179.6
total_r = 1179.6;
// `cap' side length = sqrt(666.66^2+100^2) = 674.05
cap_r = 674.05;
// capped side length = total minus top = 1179.6- cap = 1179.6-674.05 = 505.5

// 2Pi*r of inner circle is 628.3
// 628.3/2*pi*674.05 = 0.148 radians = 53.47 degrees

degrees = 53.47;

// for smaller test cut
testdegrees = 26;


module capped_cone() {
    intersection(){
        difference(){
            // circle(r=total_r);
            circle(r=cap_r+250);
            circle(r=cap_r);
            rays();
            crosses();
        }
        slice();
        
    }
    
}



module slice() {
    intersection(){
        square(size=[total_r, total_r]);
        rotate([0, 0, 90-testdegrees]) { 
            square(size=[total_r, total_r]);
   
        }
    }
    
}

mindist = cap_r+((total_r-cap_r-500)/2);
mindist_in = cap_r;
mindist_out = cap_r+(total_r-cap_r-500);
betweendist = 20;

// set_one_dists = [mindist_out, 
//                 mindist+125+betweendist/2, 
//                 mindist+250+betweendist/2, 
//                 mindist+375+betweendist/2];

// set_two_dists = [mindist_in, 
//                 mindist+62.5+betweendist/2, 
//                 mindist+187.5+betweendist/2, 
//                 mindist+312.5+betweendist/2,
//                 mindist+437.5+betweendist/2];

linewidth = 1;

sidelen = total_r-cap_r;
raylen = 100;

module rayline(mdist) {
    for (i=[0:1:6]) {
        translate([0, mdist+i*(raylen+betweendist), 0]) {
        // square(size=[linewidth, raylen]);
        debug_bezier([[0,0],[0,0],[0,raylen],[0,raylen]]);


        }
    }
}



// module ray_set_2() {
//     for (l=[set_two_dists[0], set_two_dists[4]]) {
//         translate([0, l, 0]) {
//            square(size=[linewidth, 62.5-betweendist]);
//         }
//     }
    
//     for (l=[set_two_dists[1],set_two_dists[2],set_two_dists[3]]) {
//         translate([0, l, 0]) {
//         square(size=[linewidth, 125-betweendist]);
//         }
//     }
// }


// tests are 60, 80, 100
nlines = 60;
angle = degrees/(nlines/4);

nverticalwires = 17*4;
nhorizontalwires = 29;
crossangle = degrees/nverticalwires;

wireoffsettop = 20;
wireoffsetbottom = 20;

crossdistvert = (505.5- wireoffsettop -wireoffsetbottom)/(nhorizontalwires+1);

// cross = square(size=[2, 0.8], center=true);


module crosses() {
    for (d=[1:1:nhorizontalwires]){
        for (i=[crossangle/2:crossangle:degrees]){
            rotate([0, 0, -i])  translate([0, cap_r+wireoffsettop+d*crossdistvert, 0]) rotate([0, 0, -45])  square(size=[2, 0.9], center=true);;
        }
    }

}

offset = (raylen+(betweendist/2))/4;
angleoffset = 0.4;
module rays() {
    for (i=[angleoffset:angle:degrees]) {
        rotate([0, 0, -i]) {
            // ray_set_1();
            rayline(mindist);
        }
    }
    for (i=[angleoffset+(angle/4):angle:degrees]) {
        rotate([0, 0, -i]) {
            // ray_set_1();
            rayline(mindist-offset*2);
        }
    }
    for (i=[angleoffset+(angle/2):angle:degrees]) {
        rotate([0, 0, -i]) {
            rayline(mindist-offset);
        }
    
    }
    for (i=[angleoffset+((angle/4)*3):angle:degrees]) {
        rotate([0, 0, -i]) {
            rayline(mindist-offset*3);
        }
    
    }
}


capped_cone();

// translate([0, 650, 0])  square(size=[10, 10]);
// translate([0, 630, 0])  text("10mm");