RMSD_MAX = 500
RMSF_MAX = 500
END_SAMPLE_FROM = 1000
TOTAL_FRAMES = 100
HTML = r"""
<div id="viewport" style="width:100%; height:100%;"></div>
<input id="trajectory-element" type="file">
<input id="structure-element" type="file">
"""

CSS = r"""
* { margin: 0; padding: 0; }
html, body { width: 100%; height: 100%; overflow: hidden; }
"""

JS = r"""

var traj;
var stage;
var player;


function setStageView(o, stage) {
    stage.autoView();
    var m = new NGL.Matrix4().makeRotationY(Math.PI);
    stage.viewerControls.applyMatrix(m);
    return o;
}

async function loadTraj(o) {
    let trajectoryBlob = document.getElementById("trajectory-element").files[0];
    let x = await NGL.autoLoad( trajectoryBlob, { ext: "xtc" } );
    return o.addTrajectory(x);
}

async function loadStage() {
    let structureBlob = document.getElementById("structure-element").files[0];
    let isCG = window.isCoarseGrained || false;
    stage = new NGL.Stage("viewport");
    stage.setParameters({ backgroundColor: "white" });
    window.addEventListener("resize", function( event ){
        stage.handleResize();
    }, false);

    await stage.loadFile(structureBlob, {defaultRepresentation: !isCG}).then((o) => {
        if (isCG) {
            o.addRepresentation("spacefill", { radiusScale: 3.0 });
        }
        setStageView(o, stage);
    });

    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));

    return true;
}

function setFrame(i) {
    player.setParameters({start: i, end: i, step: 1, mode: 'once'});
    player.play();
    player.stop();
    return true;
}

"""

AA3TO1 = {
    # standard residues
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "PYL": "O",
    "SEC": "U",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    # nonstandard
    "HIE": "H",
    "HID": "H",
    "HIP": "H",
    "HSE": "H",
    "HSD": "H",
    "HSP": "H",
    "MSE": "M",
}

SOFTWARE_TO_FILE_EXT = {
    "acemd": [".psf", ".xtc", ".pdb"],
    "acemd3": [".psf", ".xtc", ".pdb"],
    "amber": [".prmtop", ".nc", ".inpcrd", ".mdcrd"],
    "charmm": [".psf", ".rtf", ".dcd", ".crd"],
    "desmond": [".cms", ".dtr"],
    "gromacs": [".gro", ".top", ".tpr", ".xtc", ".trr", ".mdp", ".pdb"],
    "lammps": [".data", ".dump"],
    "namd": [".psf", ".pdb", ".dcd"],
}
