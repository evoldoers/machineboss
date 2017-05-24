// Ian Holmes, 2017
// A slimmed-down port of David Matei's fast5.hpp which can be found here:
// https://github.com/mateidavid/fast5/blob/master/src/fast5.hpp

var extend = require('extend')

var hdf5_module = require('hdf5')
var hdf5_globals = require('hdf5/lib/globals')

var hdf5 = hdf5_module.hdf5
var h5lt = hdf5_module.h5lt
var h5tb = hdf5_module.h5tb

var Access = hdf5_globals.Access
var H5OType = hdf5_globals.H5OType

// Constructor

var File = function (filename) {
  extend (this,
          { channel_id_params: {},
            raw_samples_read_names: [],
            event_detection_groups: [],
            event_detection_read_names: {},
            basecall_groups: [],
            basecall_group_descriptions: {},
            basecall_strand_groups: [[], [], []] })
  
  if (filename)
    this.open (filename)
}

// HDF5 helpers

File.prototype.path_parent_child = function (path) {
  var path_split = path.split('/'), child = path_split.pop()
  return { parent: path_split.join('/'),
           child: child }
}

File.prototype.get_object_type = function (path) {
  var pc = this.path_parent_child (path)
  if (/\//.test(pc.parent) && !this.group_exists(pc.parent))
    return H5OType.H5O_TYPE_UNKNOWN
  var parent = pc.parent.length ? this.file.openGroup(pc.parent) : this.file
  var isChild = {}
  parent.getMemberNames().forEach (function (child) { isChild[child] = true })
  if (!isChild[pc.child])
    return H5OType.H5O_TYPE_UNKNOWN
  return parent.getChildType (pc.child)
}

File.prototype.group_exists = function (path) {
  return this.get_object_type(path) === H5OType.H5O_TYPE_GROUP
}

File.prototype.dataset_exists = function (path) {
  return this.get_object_type(path) === H5OType.H5O_TYPE_DATASET
}

// Fast5 internal paths

File.prototype.file_version_path = function() { return "/file_version"; }
File.prototype.channel_id_path = function()   { return "/UniqueGlobalKey/channel_id"; }
File.prototype.tracking_id_path = function()  { return "/UniqueGlobalKey/tracking_id"; }
File.prototype.sequences_path = function()    { return "/Sequences/Meta"; }
File.prototype.raw_samples_root_path = function() { return "/Raw/Reads"; }
File.prototype.raw_samples_params_path = function(rn)
{
  return this.raw_samples_root_path() + "/" + rn;
}
File.prototype.raw_samples_path = function(rn)
{
  return this.raw_samples_root_path() + "/" + rn + "/Signal";
}
File.prototype.event_detection_root_path = function() { return "/Analyses"; }
File.prototype.event_detection_group_prefix = function() { return "EventDetection_"; }
File.prototype.event_detection_group_path = function (gr)
{
  return this.event_detection_root_path() + "/" + this.event_detection_group_prefix() + gr;
}
File.prototype.event_detection_events_params_path = function (gr, rn)
{
  return this.event_detection_group_path(gr) + "/Reads/" + rn;
}
File.prototype.event_detection_events_path = function (gr, rn)
{
  return this.event_detection_group_path(gr) + "/Reads/" + rn + "/Events";
}
File.prototype.basecall_root_path = function() { return "/Analyses"; }
File.prototype.basecall_group_prefix = function() { return "Basecall_"; }
var _strand_name = [ "template", "complement", "2D" ]
File.prototype.strand_name = function(st)
{
  return _strand_name[st];
}
File.prototype.basecall_strand_subgroup = function(st)
{
  return "BaseCalled_" + this.strand_name(st);
}
File.prototype.basecall_group_path = function (gr)
{
  return this.basecall_root_path() + "/" + this.basecall_group_prefix() + gr;
}
File.prototype.basecall_strand_group_path = function (gr, st)
{
  return this.basecall_group_path(gr) + "/" + this.basecall_strand_subgroup(st);
}
File.prototype.basecall_log_path = function (gr)
{
  return this.basecall_group_path(gr) + "/Log";
}
File.prototype.basecall_fastq_path = function (gr, st)
{
  return this.basecall_strand_group_path(gr, st) + "/Fastq";
}
File.prototype.basecall_events_path = function (gr, st)
{
  return this.basecall_strand_group_path(gr, st) + "/Events";
}
File.prototype.basecall_alignment_path = function(gr)
{
  return this.basecall_strand_group_path(gr, 2) + "/Alignment";
}
File.prototype.basecall_config_path = function(gr)
{
  return this.basecall_group_path(gr) + "/Configuration";
}
File.prototype.basecall_summary_path = function(gr)
{
  return this.basecall_group_path(gr) + "/Summary";
}

// Cache updaters

File.prototype.open = function (filename) {
  this.file = new hdf5.File (filename, Access.ACC_RDONLY)
  this.reload()
}

File.prototype.reload = function() {
  this.load_channel_id_params()
  this.load_raw_samples_read_names()
  this.load_event_detection_groups()
  this.load_basecall_groups()
}

File.prototype.load_channel_id_params = function() {
  this.channel_id_params = this.file.getDatasetAttributes (this.channel_id_path())
}

File.prototype.load_raw_samples_read_names = function() {
  if (this.group_exists(this.raw_samples_root_path())) {
    var group = this.file.openGroup (this.raw_samples_root_path())
    this.raw_samples_read_names = group.getMemberNamesByCreationOrder()
  }
}

File.prototype.load_event_detection_groups = function() {
  var fast5 = this
  if (this.group_exists(this.event_detection_root_path())) {
    var ed_gr_prefix = this.event_detection_group_prefix()
    var group = this.file.openGroup (this.event_detection_root_path())
    var ed_group_names = group.getMemberNamesByCreationOrder()
    this.event_detection_groups = ed_group_names.filter (function (ed_group_name) {
      return ed_group_name.substr(0,ed_gr_prefix.length) === ed_gr_prefix
    }).map (function (ed_group_name) {
      return ed_group_name.substr (ed_gr_prefix.length)
    })
    this.event_detection_groups.forEach (function (ed_gr_suffix) {
      var ed_group = group.openGroup (ed_gr_prefix + ed_gr_suffix + '/Reads')
      fast5.event_detection_read_names[ed_gr_suffix] = ed_group.getMemberNamesByCreationOrder()
    })
  }
}

File.prototype.load_basecall_groups = function() {
  var fast5 = this
  if (this.group_exists (this.basecall_root_path())) {
    var bc_gr_prefix = this.basecall_group_prefix()
    var group = this.file.openGroup (this.basecall_root_path())
    var bc_group_names = group.getMemberNamesByCreationOrder()
    this.basecall_groups = bc_group_names.filter (function (bc_group_name) {
      return bc_group_name.substr(0,bc_gr_prefix.length) === bc_gr_prefix
    }).map (function (bc_group_name) {
      return bc_group_name.substr (bc_gr_prefix.length)
    })
    this.basecall_groups.forEach (function (gr) {
      var desc = { name: "?",
                   version: "?",
                   have_subgroup: [false, false, false],
                   have_fastq: [false, false, false],
                   have_events: [false, false, false] }
      var attr = fast5.file.getDatasetAttributes (fast5.basecall_group_path(gr))
      if (attr.name === "ONT Sequencing Workflow") {
        desc.name = "metrichor";
        desc.version = (attr["chimaera version"] || "?") + "+" +
          (attr["dragonet version"] || "?");
      } else if (attr.name === "MinKNOW-Live-Basecalling") {
        desc.name = "minknow";
        desc.version = attr.version || "?"
      } else if (attr.name == "ONT Albacore Sequencing Software") {
        desc.name = "albacore";
        desc.version = attr.version || "?"
      }
      for (var st = 0; st < 3; ++st) {
        if (fast5.group_exists (fast5.basecall_strand_group_path(gr, st))) {
          desc.have_subgroup[st] = true
          fast5.basecall_strand_groups[st].push(gr)
          desc.have_fastq[st] = fast5.dataset_exists (fast5.basecall_fastq_path(gr, st))
          desc.have_events[st] = fast5.dataset_exists (fast5.basecall_events_path(gr, st))
          if (st === 0)
            desc.ed_gr = gr
          if (st === 2)
            desc.have_alignment = fast5.have_basecall_alignment(gr)
        }
      }
      if (desc.have_subgroup[0] || desc.have_subgroup[1])
        desc.bc_1d_gr = gr
      else if (desc.have_subgroup[2])
        desc.bc_1d_gr = gr
      
      fast5.basecall_group_descriptions[gr] = desc
    })
  }
}

File.prototype.detect_basecall_event_detection_group = function(gr) {
  var bc_params = this.get_basecall_params(gr)
  var pref = this.event_detection_root_path().substr(1) + "/" + this.event_detection_group_prefix();
  var tmp = bc_params["event_detection"]
  if (tmp) {
    var pref = this.event_detection_root_path().substr(1) + "/" + this.event_detection_group_prefix();
    if (tmp.substr(0, pref.length) === pref) {
      var ed_gr = tmp.substr(pref.length);
      if (this.have_event_detection_group(ed_gr))
        return ed_gr;
    }
  }
  return ""
}

File.prototype.get_basecall_params = function(gr) {
  return this.file.getDatasetAttributes (this.basecall_group_path(gr))
}

File.prototype.have_basecall_alignment = function(gr) {
  return this.dataset_exists (this.basecall_alignment_path(gr))
}

File.prototype.have_event_detection_group = function(ed_gr) {
  return event_detection_read_names[ed_gr]
}

File.prototype.fill_raw_samples_read_name = function(rn) {
  return (rn.length || !this.raw_samples_read_names.length) ? rn : this.raw_samples_read_names[0]
}

File.prototype.fill_event_detection_group = function(gr) {
  return (gr.length || !this.event_detection_groups.length) ? gr : this.event_detection_groups[0]
}

File.prototype.fill_event_detection_read_name = function(gr, rn) {
  return (rn.length || !this.event_detection_read_names[gr] || !this.event_detection_read_names[gr].length) ? rn : this.event_detection_read_names[gr][0]
}

File.prototype.fill_basecall_group = function(st, gr) {
  return (gr.length || !this.basecall_strand_groups[st]) ? gr : this.basecall_strand_groups[st][0]
}

File.prototype.fill_basecall_1d_group = function(st, gr) {
  var _gr = this.fill_basecall_group(st, gr);
  return this.get_basecall_1d_group(_gr);
}

// Access /file_version
// Access /UniqueGlobalKey/channel_id
// Access /UniqueGlobalKey/tracking_id
// Access /Sequences
// Access Raw Samples
// Access EventDetection groups
// Access EventDetection events
// Access Basecall groups
// Access Basecall group params
// Access Basecall group log
// Access Basecall fastq
// Access Basecall events
// Static helpers


exports.File = File
