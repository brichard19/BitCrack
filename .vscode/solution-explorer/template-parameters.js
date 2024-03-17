const path = require("path");

module.exports = function(filename, projectPath, folderPath, xml) {
    let namespace;
    xml.elements.some(e => {
        if (e.name === 'PropertyGroup' && e.elements) {
            const rootNamespace = e.elements.find(p => p.name === 'RootNamespace');
            if (rootNamespace && rootNamespace.elements && rootNamespace.elements[0]) {
                namespace = rootNamespace.elements[0].text;
                return true;
            }
        }
    });

    if (!namespace && projectPath) {
        namespace = path.basename(projectPath, path.extname(projectPath));
        if (folderPath) {
            namespace += "." + folderPath.replace(path.dirname(projectPath), "").substring(1).replace(/[\\\/]/g, ".");
        }
        namespace = namespace.replace(/[\\\-]/g, "_");
    }

    if (!namespace) {
        namespace = "Unknown";
    }

    return {
        namespace: namespace,
        name: path.basename(filename, path.extname(filename))
    }
};
