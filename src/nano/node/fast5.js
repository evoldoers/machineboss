// Ian Holmes, 2017
// A simplified port of David Matei's fast5.hpp which can be found here:
// https://github.com/mateidavid/fast5/blob/master/src/fast5.hpp

var hdf5 = require('hdf5').hdf5
var extend = require('extend')

// constructor

var File = function (filename) {
  extend (this,
          { channel_id_params: {},
            raw_samples_read_names: [],
            event_detection_groups: [],
            event_detection_read_names: {},
            basecall_groups: [],
            basecall_group_descriptions: {},
            basecall_strand_groups: [null, null, null] })
  
  if (filename)
    this.open (filename)
}

// HDF5 string constants

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
File.prototype.raw_samples_pack_path = function(rn)
{
  return this.raw_samples_path(rn) + "_Pack";
}
File.prototype.raw_samples_params_pack_path = function(rn)
{
  return this.raw_samples_pack_path(rn) + "/params";
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
File.prototype.event_detection_events_pack_path = function (gr, rn)
{
  return this.event_detection_events_path(gr, rn) + "_Pack";
}
File.prototype.event_detection_events_params_pack_path = function (gr, rn)
{
  return this.event_detection_events_pack_path(gr, rn) + "/params";
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
File.prototype.basecall_fastq_pack_path = function (gr, st)
{
  return this.basecall_fastq_path(gr, st) + "_Pack";
}
File.prototype.basecall_model_path = function (gr, st)
{
  return this.basecall_strand_group_path(gr, st) + "/Model";
}
File.prototype.basecall_model_file_path = function (gr, st)
{
  return this.basecall_group_path(gr) + "/Summary/basecall_1d_" + this.strand_name(st) + "/model_file";
}
File.prototype.basecall_events_path = function (gr, st)
{
  return this.basecall_strand_group_path(gr, st) + "/Events";
}
File.prototype.basecall_events_pack_path = function (gr, st)
{
  return this.basecall_events_path(gr, st) + "_Pack";
}
File.prototype.basecall_events_params_pack_path = function (gr, st)
{
  return this.basecall_events_pack_path(gr, st) + "/params";
}
File.prototype.basecall_alignment_path = function(gr)
{
  return this.basecall_strand_group_path(gr, 2) + "/Alignment";
}
File.prototype.basecall_alignment_pack_path = function(gr)
{
  return this.basecall_alignment_path(gr) + "_Pack";
}
File.prototype.basecall_config_path = function(gr)
{
  return this.basecall_group_path(gr) + "/Configuration";
}
File.prototype.basecall_summary_path = function(gr)
{
  return this.basecall_group_path(gr) + "/Summary";
}

// methods

File.prototype.open = function (filename) {
  var Access = require('hdf5/lib/globals').Access
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
  var group = this.file.openGroup (this.raw_samples_root_path())
  this.raw_samples_read_names = group.getMemberNamesByCreationOrder()
}

File.prototype.load_event_detection_groups = function() {
  var ed_gr_prefix = this.event_detection_group_prefix()
  var group = this.file.openGroup (this.event_detection_root_path())
  var ed_group_names = group.getMemberNamesByCreationOrder()
  this.event_detection_groups = ed_group_names.filter (function (ed_group_name) {
    console.log(ed_group_name,ed_gr_prefix)
    return ed_group_name.substr(0,ed_gr_prefix.length) === ed_gr_prefix
  }).map (function (ed_group_name) {
    return ed_group_name.substr (ed_gr_prefix.length)
  })
  this.event_detection_read_names = this.event_detection_groups.map (function (ed_gr_suffix) {
    var ed_group = group.openGroup (ed_gr_prefix + ed_gr_suffix + '/Reads')
    return ed_group.getMemberNamesByCreationOrder()
  })
}

File.prototype.load_basecall_groups = function() {
  var bc_gr_prefix = this.basecall_group_prefix()
  var group = this.file.openGroup (this.basecall_root_path())
  var bc_group_names = group.getMemberNamesByCreationOrder()
  this.basecall_groups = bc_group_names.filter (function (bc_group_name) {
    return bc_group_name.substr(0,bc_gr_prefix.length) === bc_gr_prefix
  }).map (function (bc_group_name) {
    return bc_group_name.substr (bc_gr_prefix.length)
  })
  // TO BE CONTINUED
}

exports.File = File
